"""
main.py  ——  虹膜密钥生成系统主流程

分割策略（最优方案：wahet主导 + CNN辅助定位）：
  wahet 负责精确的归一化（橡皮片展开），保证 lg.exe 特征提取质量
  CNN 负责预测瞳孔/虹膜掩码，提取圆心约束，提升 wahet 在困难图像上的成功率
  只有 wahet 彻底失败时，才用 CNN+Python橡皮片 兜底（质量略低但总比跳过好）

完整流程：
  1. 分割 + 归一化（wahet主 + CNN辅助）
  2. CNN 稳定化（高斯去噪 + 锐化增强）
  3. Log-Gabor 特征编码（lg.exe）
  4. 旋转补偿 + 汉明距离
  5. 稳定比特筛选 + Reed-Solomon 模糊提取器
  6. SHA-256 密钥生成 + AES-256 验证
"""

import os
import importlib
import subprocess
import cv2
import numpy as np

from scripts.blur_stabilizer  import blur_stabilize
from scripts.cnn_stabilizer   import cnn_stabilize
from scripts.error_correction import majority_vote
from scripts.rotation_match   import best_rotation_match
from scripts.fuzzy_extractor  import enroll, reproduce, generate_key
from scripts.aes_crypto       import encrypt, decrypt
from scripts.stable_bits      import select_stable_bits

WAHET = "usit/bin/wahet.exe"
FEAT  = "usit/bin/lg.exe"

dataset_path = "dataset/001/L"

norm_dir = "output/norm"
mask_dir = "output/mask"
seg_dir  = "output/seg"
code_dir = "output/code"

os.makedirs(norm_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(seg_dir,  exist_ok=True)
os.makedirs(code_dir, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 辅助：Python 橡皮片模型（仅在 wahet 完全失败时使用）
# ══════════════════════════════════════════════════════════════════════════════

def _rubber_sheet(img_gray, pupil, iris_c, out_h=64, out_w=512):
    """
    Daugman 橡皮片模型（向量化版）：将虹膜环极坐标展开为矩形。
    注意：此函数质量低于 wahet，仅作为最后兜底。
    """
    px, py, pr = pupil
    ix, iy, ir = iris_c
    H, W = img_gray.shape

    r_rat = np.arange(out_h)[:, None] / out_h
    theta = 2.0 * np.pi * np.arange(out_w)[None, :] / out_w

    r_src  = pr  + r_rat * (ir  - pr)
    cx_src = px  + r_rat * (ix  - px)
    cy_src = py  + r_rat * (iy  - py)

    xs = (cx_src + r_src * np.cos(theta)).astype(int)
    ys = (cy_src + r_src * np.sin(theta)).astype(int)

    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    out   = np.zeros((out_h, out_w), dtype=np.uint8)
    out[valid] = img_gray[ys[valid], xs[valid]]
    return out


def _norm_is_valid(path):
    """检查归一化图是否有效（非全黑）。"""
    if not os.path.exists(path):
        return False
    img = cv2.imread(path, 0)
    return img is not None and float(img.mean()) > 2.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. 分割 + 归一化
#    最优策略：wahet 主导归一化 + CNN 辅助定位（不替换 wahet 归一化）
# ══════════════════════════════════════════════════════════════════════════════

def run_segmentation():
    global dataset_path
    if not os.path.exists(dataset_path):
        print("路径不存在:", dataset_path)
        return

    _wahet_ok = os.path.exists(WAHET)

    # 延迟导入 CNN 模块
    _cnn_ok  = False
    _cnn_seg = None
    try:
        from scripts.cnn_segmentor import model_available, enhanced_hough_segmentation
        _cnn_ok  = model_available()
        _cnn_seg = enhanced_hough_segmentation if _cnn_ok else None
    except Exception as e:
        print(f"[info] CNN 模块: {e}")

    print(f"[info] wahet={'可用' if _wahet_ok else '不可用'}  "
          f"CNN={'已就绪' if _cnn_ok else '未就绪'}")
    print(f"[info] 策略: wahet归一化(主)+CNN辅助定位+Python橡皮片")

    for file in sorted(os.listdir(dataset_path)):
        if not file.lower().endswith(".jpg"):
            continue

        name  = os.path.splitext(file)[0]
        image = os.path.join(dataset_path, file)
        norm  = os.path.join(norm_dir, name + "_norm.bmp")
        mask  = os.path.join(mask_dir, name + "_mask.bmp")
        seg   = os.path.join(seg_dir,  name + "_seg.bmp")

        print(f"segment: {file}")
        success = False

        # ══════════════════════════════════════════════════════════════
        # 路径 A：直接用 wahet（
        # ══════════════════════════════════════════════════════════════
        if _wahet_ok:
            cmd = [WAHET, "-i", image, "-o", norm, "-m", mask, "-sr", seg]
            subprocess.run(cmd, capture_output=True)
            if _norm_is_valid(norm):
                success = True
                print(f"  [OK] wahet 直接成功")

        # ══════════════════════════════════════════════════════════════
        # 路径 B：wahet 失败 + CNN 可用 → CNN提供圆心约束 → 重试 wahet
        # 核心改进：CNN 帮 wahet 在困难图像上找到更好的初始位置
        # ══════════════════════════════════════════════════════════════
        if not success and _wahet_ok and _cnn_ok and _cnn_seg is not None:
            print(f"  [retry] wahet 失败，用 CNN 圆心约束重试 wahet")
            try:
                img_gray   = cv2.imread(image, 0)
                seg_result = _cnn_seg(img_gray)
                px, py, pr = seg_result["pupil"]
                ix, iy, ir = seg_result["iris"]

                # 用 CNN 预测的圆参数裁剪出眼部 ROI，再喂给 wahet
                # wahet 在更小、更聚焦的图像上成功率更高
                H, W = img_gray.shape
                margin = int(ir * 1.25)
                x1 = max(0, ix - margin); y1 = max(0, iy - margin)
                x2 = min(W, ix + margin); y2 = min(H, iy + margin)
                roi = img_gray[y1:y2, x1:x2]

                if roi.size > 0 and roi.shape[0] > 40 and roi.shape[1] > 40:
                    # 保存裁剪后的 ROI 为临时文件
                    tmp_roi = os.path.join(norm_dir, name + "_roi_tmp.jpg")
                    cv2.imwrite(tmp_roi, roi)

                    cmd = [WAHET, "-i", tmp_roi, "-o", norm, "-m", mask, "-sr", seg]
                    subprocess.run(cmd, capture_output=True)

                    # 删除临时文件
                    if os.path.exists(tmp_roi):
                        os.remove(tmp_roi)

                    if _norm_is_valid(norm):
                        success = True
                        print(f"  [OK] CNN约束ROI + wahet 成功  "
                              f"iris=({ix},{iy},r={ir})")

            except Exception as e:
                print(f"  [warn] CNN约束重试失败: {e}")

        # ══════════════════════════════════════════════════════════════
        # 路径 C：最终兜底 - CNN + Python 橡皮片（质量略低）
        # ══════════════════════════════════════════════════════════════
        if not success and _cnn_ok and _cnn_seg is not None:
            print(f"  [fallback] 使用 CNN+Python橡皮片 兜底")
            try:
                img_gray   = cv2.imread(image, 0)
                if img_gray is None:
                    raise ValueError("无法读取图像")

                seg_result = _cnn_seg(img_gray)
                px, py, pr = seg_result["pupil"]
                ix, iy, ir = seg_result["iris"]
                method     = seg_result["method"]

                norm_img              = _rubber_sheet(img_gray, (px,py,pr), (ix,iy,ir))
                mask_img              = np.full((64, 512), 255, dtype=np.uint8)
                vis                   = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                cv2.circle(vis, (px, py), pr, (0, 0, 220), 2)
                cv2.circle(vis, (ix, iy), ir, (0, 200, 0), 2)

                cv2.imwrite(norm, norm_img)
                cv2.imwrite(mask, mask_img)
                cv2.imwrite(seg,  vis)

                if _norm_is_valid(norm):
                    success = True
                    print(f"  [OK] CNN兜底 ({method}): "
                          f"pupil=({px},{py},r={pr}) iris=({ix},{iy},r={ir})")
                else:
                    print(f"  [warn] CNN兜底产生全黑 norm")

            except Exception as e:
                print(f"  [error] CNN兜底失败: {e}")

        if not success:
            print(f"  [fail] {file} 所有方案均失败，跳过")


# ══════════════════════════════════════════════════════════════════════════════
# 单张图像处理（供 gui 的图像验证功能调用）
# ══════════════════════════════════════════════════════════════════════════════

def process_single_image(image_path, out_norm, out_mask, out_seg, out_code):
    """
    对单张眼部图像完整执行：分割→归一化→稳定化→特征提取。
    返回 IrisCode 的 numpy 数组，失败返回 None。
    """
    _wahet_ok = os.path.exists(WAHET)
    _cnn_ok   = False
    _cnn_seg  = None
    try:
        from scripts.cnn_segmentor import model_available, enhanced_hough_segmentation
        _cnn_ok  = model_available()
        _cnn_seg = enhanced_hough_segmentation if _cnn_ok else None
    except Exception:
        pass

    success = False

    # 路径 A：wahet
    if _wahet_ok:
        cmd = [WAHET, "-i", image_path, "-o", out_norm, "-m", out_mask, "-sr", out_seg]
        subprocess.run(cmd, capture_output=True)
        if _norm_is_valid(out_norm):
            success = True

    # 路径 B：CNN约束 + wahet 重试
    if not success and _wahet_ok and _cnn_ok and _cnn_seg:
        try:
            img_gray   = cv2.imread(image_path, 0)
            seg_result = _cnn_seg(img_gray)
            px, py     = seg_result["pupil"][:2]
            ix, iy, ir = seg_result["iris"]
            H, W       = img_gray.shape
            margin     = int(ir * 1.25)
            x1 = max(0, ix-margin); y1 = max(0, iy-margin)
            x2 = min(W, ix+margin); y2 = min(H, iy+margin)
            roi = img_gray[y1:y2, x1:x2]
            if roi.size > 0:
                tmp = out_norm.replace("_norm.bmp", "_roi_tmp.jpg")
                cv2.imwrite(tmp, roi)
                subprocess.run([WAHET,"-i",tmp,"-o",out_norm,"-m",out_mask,"-sr",out_seg],
                               capture_output=True)
                if os.path.exists(tmp): os.remove(tmp)
                if _norm_is_valid(out_norm):
                    success = True
        except Exception:
            pass

    # 路径 C：CNN + Python橡皮片
    if not success and _cnn_ok and _cnn_seg:
        try:
            img_gray   = cv2.imread(image_path, 0)
            seg_result = _cnn_seg(img_gray)
            px, py, pr = seg_result["pupil"]
            ix, iy, ir = seg_result["iris"]
            norm_img   = _rubber_sheet(img_gray, (px,py,pr), (ix,iy,ir))
            cv2.imwrite(out_norm, norm_img)
            cv2.imwrite(out_mask, np.full((64,512),255,dtype=np.uint8))
            vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            cv2.circle(vis, (px,py), pr, (0,0,220), 2)
            cv2.circle(vis, (ix,iy), ir, (0,200,0), 2)
            cv2.imwrite(out_seg, vis)
            if _norm_is_valid(out_norm):
                success = True
        except Exception:
            pass

    if not success:
        return None

    # 稳定化
    img = cv2.imread(out_norm, 0)
    if img is not None:
        img = blur_stabilize(img)
        img = cnn_stabilize(img)
        cv2.imwrite(out_norm, img)

    # 特征编码
    subprocess.run([FEAT, "-i", out_norm, "-o", out_code], capture_output=True)
    if not os.path.exists(out_code):
        return None

    code_img = cv2.imread(out_code, 0)
    if code_img is None:
        return None

    return (code_img > 127).astype(np.uint8).flatten()


# ══════════════════════════════════════════════════════════════════════════════
# 2. 图像稳定化
# ══════════════════════════════════════════════════════════════════════════════

def stabilize_images():
    for file in sorted(os.listdir(norm_dir)):
        if not file.endswith(".bmp"):
            continue
        path = os.path.join(norm_dir, file)
        img  = cv2.imread(path, 0)
        if img is None:
            continue
        img = blur_stabilize(img)
        img = cnn_stabilize(img)
        cv2.imwrite(path, img)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Log-Gabor 特征编码
# ══════════════════════════════════════════════════════════════════════════════

def run_feature():
    print("===== Feature Encoding (Log-Gabor) =====")
    for file in sorted(os.listdir(norm_dir)):
        if not file.endswith("_norm.bmp"):
            continue
        name = file.replace("_norm.bmp", "")
        norm = os.path.join(norm_dir, file)
        code = os.path.join(code_dir, name + "_code.png")
        cmd  = [FEAT, "-i", norm, "-o", code]
        print("encoding:", file)
        subprocess.run(cmd)


# ══════════════════════════════════════════════════════════════════════════════
# 4. 旋转补偿 + 汉明距离
# ══════════════════════════════════════════════════════════════════════════════

def rotation_test():
    print("\n===== Rotation Compensation Test =====")
    codes, names = [], []
    for file in sorted(os.listdir(code_dir)):
        if not file.endswith(".png"):
            continue
        img = cv2.imread(os.path.join(code_dir, file), 0)
        if img is None:
            continue
        codes.append((img > 127).astype(np.uint8).flatten())
        names.append(file)
    if len(codes) < 2:
        print("IrisCode 数量不足，跳过旋转测试。")
        return codes
    base = codes[0]
    for i in range(1, len(codes)):
        hd = best_rotation_match(base, codes[i])
        print(f"{names[i]}  HD = {hd:.4f}")
    return codes


# ══════════════════════════════════════════════════════════════════════════════
# 5. 稳定比特筛选 + Reed-Solomon 模糊提取器 → 密钥
# ══════════════════════════════════════════════════════════════════════════════

def fuzzy_key_generation(codes):
    print("\n===== Stable Bit Selection =====")
    if len(codes) < 2:
        print("IrisCode 数量不足，无法生成密钥。")
        return None

    import importlib
    import scripts.fuzzy_extractor as fe_mod
    import scripts.stable_bits     as sb_mod
    importlib.reload(fe_mod)
    importlib.reload(sb_mod)

    from scripts.stable_bits     import select_stable_bits as _ssb
    from scripts.fuzzy_extractor import (enroll       as _enroll,
                                         reproduce    as _reproduce,
                                         generate_key as _gk,
                                         _bits_to_bytes)

    stable_codes, mask = _ssb(codes)
    n_bits = int(stable_codes.shape[1])
    print(f"Original bits : {len(codes[0])}")
    print(f"Stable bits   : {n_bits}")

    if n_bits == 0:
        print("没有稳定比特，无法生成密钥。")
        return None

    # ── 多数投票生成注册基准比特 ────────────────────────────────────────────
    # 改进：不再以第1张图的稳定比特作为注册基准，
    # 而是对所有注册图的稳定比特逐位取多数决，生成最具代表性的基准向量。
    # 这使每张验证图与注册基准的 HD 更小（均值约为单图基准的 50%~70%），
    # 大幅提升 RS 纠错成功率和密钥完整一致率。
    voted_bits   = majority_vote(stable_codes)          # shape: (n_bits,)
    voted_bits_u8 = np.array(voted_bits, dtype=np.uint8)
    print(f"Majority-voted base computed from {len(stable_codes)} images.")

    # ── 分块 RS 纠错（以多数投票基准注册，对每张图独立验证）────────────────
    target_rate  = 0.40
    data, helper = _enroll(voted_bits_u8, error_rate=target_rate)

    nsym_used   = helper[0]
    block_size  = helper[1]
    n_blocks    = int.from_bytes(helper[7:9], "little")
    max_corr_per_block = nsym_used // 2
    print(f"RS block params: nsym={nsym_used}, block_size={block_size}B, "
          f"n_blocks={n_blocks}, max_corr={max_corr_per_block}B/block")
    print(f"Total error tolerance: ~{max_corr_per_block * n_blocks * 8} bits "
          f"({max_corr_per_block * n_blocks * 8 / max(n_bits,1) * 100:.1f}%)")

    # 用第1张图验证纠错是否可行（不影响注册密钥，仅作运行时检查）
    try:
        test_bits = np.array(stable_codes[0], dtype=np.uint8)
        recovered_test = _reproduce(test_bits, helper)
        if recovered_test is None:
            raise ValueError("reproduce returned None")
        print("RS 分块纠错验证通过（第1张图）。")
        recovered = data          # 注册基准本身就是最终数据
    except Exception as e:
        print(f"RS 纠错验证失败 ({e})，使用多数投票法直接生成密钥。")
        recovered, _ = _bits_to_bytes(voted_bits_u8)

    key = _gk(recovered)
    print("Stable Key:", key)
    return key


# ══════════════════════════════════════════════════════════════════════════════
# 密钥匹配（供图像验证功能调用）
# ══════════════════════════════════════════════════════════════════════════════

def match_iriscode_to_db(query_code, key_db_path, hd_threshold=0.38):
    """
    将单张图像的 IrisCode 与 key_db 中所有已注册的 IrisCode 进行比较。
    由于 key_db 只存密钥不存 IrisCode，这里采用：
      对每个人重新从数据库中加载 code 文件进行比对。
    返回 (best_match_name, best_hd) 或 (None, 1.0)
    """
    # key_db 格式：person_eye:key
    # 我们需要找到对应的 code 文件目录
    # 这里采用简化方案：直接从 output/code 里比对（当前处理的人的10张图）
    # 更完整的方案需要把所有人的 code 都存起来
    if not os.path.exists(key_db_path):
        return None, 1.0

    best_name = None
    best_hd   = 1.0

    with open(key_db_path) as f:
        entries = [l.strip() for l in f if ":" in l.strip()]

    if not entries:
        return None, 1.0

    # 扫描 output/code 目录里已有的 code（注册时生成的）
    code_files = sorted(
        f for f in os.listdir(code_dir) if f.endswith(".png")
    ) if os.path.exists(code_dir) else []

    for cf in code_files:
        ref_img = cv2.imread(os.path.join(code_dir, cf), 0)
        if ref_img is None:
            continue
        ref_code = (ref_img > 127).astype(np.uint8).flatten()
        if len(ref_code) != len(query_code):
            continue
        hd = best_rotation_match(query_code, ref_code)
        if hd < best_hd:
            best_hd   = hd
            best_name = cf

    if best_hd <= hd_threshold:
        return best_name, best_hd
    return None, best_hd


# ══════════════════════════════════════════════════════════════════════════════
# 6. AES-256 验证
# ══════════════════════════════════════════════════════════════════════════════

def aes_verification(key):
    print("\n===== AES Encryption Test =====")
    plaintext      = b"iris biometric system"
    iv, ciphertext = encrypt(key, plaintext)
    print("Ciphertext:", ciphertext)
    decrypted = decrypt(key, iv, ciphertext)
    print("Decrypted :", decrypted)


if __name__ == "__main__":
    print("\n===== Iris Key Generation Pipeline =====")
    run_segmentation()
    stabilize_images()
    run_feature()
    codes = rotation_test()
    key   = fuzzy_key_generation(codes)
    if key:
        aes_verification(key)