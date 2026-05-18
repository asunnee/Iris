"""
prepare_training_data.py
放在项目根目录（与 main.py 同级）

功能：
  遍历 dataset/ 下所有人的左右眼图像，
  调用 wahet.exe 生成 norm.bmp 和 mask.bmp，
  从 mask.bmp 中提取瞳孔/虹膜轮廓，
  生成训练用的掩码图像，保存到 dataset/train_data/

输出目录结构：
  dataset/train_data/
    images/   原始眼部灰度图
    pupil/    瞳孔二值掩码
    iris/     虹膜环二值掩码（虹膜区域 - 瞳孔区域）

用法：
  python prepare_training_data.py
"""

import os
import subprocess
import cv2
import numpy as np

# ── 路径配置 ────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
WAHET          = os.path.join(BASE_DIR, "usit", "bin", "wahet.exe")
DATASET_ROOT   = os.path.join(BASE_DIR, "dataset")
TRAIN_DATA_DIR = os.path.join(BASE_DIR, "dataset", "train_data")
TEMP_DIR       = os.path.join(BASE_DIR, "temp_seg")

os.makedirs(TEMP_DIR, exist_ok=True)
for sub in ["images", "pupil", "iris"]:
    os.makedirs(os.path.join(TRAIN_DATA_DIR, sub), exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 核心：从 mask.bmp 提取瞳孔和虹膜圆参数
# ══════════════════════════════════════════════════════════════════════════════

def extract_circles_from_mask(mask_bmp_path, orig_img_shape):
    """
    wahet 生成的 mask.bmp 是一张二值图：
      - 黑色（0）= 被遮挡区域（眼睑/睫毛）
      - 白色（255）= 有效虹膜区域

    从 mask.bmp 用 HoughCircles 提取两个圆：
      - 小圆 = 瞳孔边界
      - 大圆 = 虹膜外边界

    返回：(pupil_circle, iris_circle)
          每个 circle = (cx, cy, r) 或 None
    """
    mask = cv2.imread(mask_bmp_path, 0)
    if mask is None:
        return None, None

    H, W = orig_img_shape

    # mask.bmp 是归一化后的尺寸（64×512），需要知道原图尺寸
    # 但我们用原图做霍夫，不用 mask 做霍夫
    # mask 只用来估算有效区域的行范围
    return None, None   # 占位，实际从原图做霍夫（见下方）


def extract_circles_from_image(img_gray):
    """
    直接对原始眼部灰度图用霍夫变换提取瞳孔和虹膜圆。
    这是最可靠的方式，与 wahet 内部逻辑一致。

    返回：(pupil, iris)  各为 (cx, cy, r) 或 None
    """
    H, W = img_gray.shape

    # 预处理：高斯模糊降噪
    blurred = cv2.GaussianBlur(img_gray, (7, 7), 1.5)

    # ── 瞳孔检测（较小、较暗的圆）──────────────────────────────────────────
    # 瞳孔半径通常是图像高度的 10%~30%
    r_min_p = max(5,  int(H * 0.08))
    r_max_p = max(10, int(H * 0.30))

    pupil = None
    circles_p = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT, dp=1,
        minDist=int(H * 0.1),
        param1=60, param2=20,
        minRadius=r_min_p,
        maxRadius=r_max_p
    )
    if circles_p is not None:
        # 选最黑的（均值最小）圆作为瞳孔
        best_c, best_val = None, 255
        for c in circles_p[0]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            # 创建掩码取圆内均值
            tmp = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(tmp, (cx, cy), r, 255, -1)
            mean_val = float(cv2.mean(img_gray, mask=tmp)[0])
            if mean_val < best_val:
                best_val = mean_val
                best_c   = (cx, cy, r)
        pupil = best_c

    # ── 虹膜检测（较大的圆）────────────────────────────────────────────────
    # 虹膜半径通常是图像高度的 30%~55%
    r_min_i = max(10, int(H * 0.28))
    r_max_i = max(20, int(H * 0.56))

    iris = None
    circles_i = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT, dp=1,
        minDist=int(H * 0.1),
        param1=50, param2=18,
        minRadius=r_min_i,
        maxRadius=r_max_i
    )
    if circles_i is not None:
        # 选最靠近图像中心的圆作为虹膜
        cx_img, cy_img = W // 2, H // 2
        best_c, best_dist = None, float("inf")
        for c in circles_i[0]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            dist = (cx - cx_img)**2 + (cy - cy_img)**2
            if dist < best_dist:
                best_dist = dist
                best_c    = (cx, cy, r)
        iris = best_c

    # ── 兜底：若某个圆检测失败，用估算值 ────────────────────────────────────
    if pupil is None:
        pupil = (W // 2, H // 2, int(H * 0.15))
        print("    [warn] 瞳孔检测失败，使用估算值")
    if iris is None:
        iris = (W // 2, H // 2, int(H * 0.42))
        print("    [warn] 虹膜检测失败，使用估算值")

    # ── 保证瞳孔在虹膜内部，且半径合理 ─────────────────────────────────────
    if pupil[2] >= iris[2]:
        # 瞳孔比虹膜大说明检测混淆了，交换
        pupil, iris = iris, pupil

    return pupil, iris


def make_iris_ring_mask(H, W, pupil, iris):
    """
    生成虹膜环掩码 = 虹膜大圆区域 - 瞳孔小圆区域。
    """
    iri_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(iri_mask, (iris[0],  iris[1]),  iris[2],  255, -1)
    cv2.circle(iri_mask, (pupil[0], pupil[1]), pupil[2],   0, -1)  # 挖掉瞳孔
    return iri_mask


def make_pupil_mask(H, W, pupil):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(mask, (pupil[0], pupil[1]), pupil[2], 255, -1)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# 验证掩码是否有意义（非全黑检查）
# ══════════════════════════════════════════════════════════════════════════════

def is_valid_mask(mask, min_white_ratio=0.005):
    """白色像素占比太低说明掩码无效。"""
    ratio = np.mean(mask > 0)
    return ratio >= min_white_ratio


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════

def generate_training_data(use_wahet=True, keep_temp=False):
    """
    use_wahet=True  : 先用 wahet 生成 norm.bmp（归一化图），保存到训练集
                      再从原图做霍夫提取圆参数生成掩码
    use_wahet=False : 跳过 wahet，直接从原图做霍夫（wahet 不可用时的备选）
    """
    if use_wahet and not os.path.exists(WAHET):
        print(f"[warn] 找不到 wahet.exe: {WAHET}")
        print("[warn] 将直接用原图生成训练数据（不经过 wahet 归一化）")
        use_wahet = False

    success = 0
    fail    = 0

    for person in sorted(os.listdir(DATASET_ROOT)):
        if person == "train_data":
            continue
        person_path = os.path.join(DATASET_ROOT, person)
        if not os.path.isdir(person_path):
            continue

        for eye in ["L", "R"]:
            eye_path = os.path.join(person_path, eye)
            if not os.path.isdir(eye_path):
                continue

            for file in sorted(os.listdir(eye_path)):
                if not file.lower().endswith((".jpg", ".png", ".bmp")):
                    continue

                img_path  = os.path.join(eye_path, file)
                stem      = os.path.splitext(file)[0]
                base_name = f"{person}_{eye}_{stem}"

                print(f"\n处理: {base_name}")

                # ── 读取原图 ─────────────────────────────────────────────
                img_orig = cv2.imread(img_path, 0)
                if img_orig is None:
                    print(f"  [skip] 无法读取图像: {img_path}")
                    fail += 1
                    continue

                H, W = img_orig.shape

                # ── 可选：用 wahet 生成归一化图（仅用于保存训练图像）────
                norm_img = img_orig   # 默认用原图
                if use_wahet:
                    tmp_norm = os.path.join(TEMP_DIR, base_name + "_norm.bmp")
                    tmp_mask = os.path.join(TEMP_DIR, base_name + "_mask.bmp")

                    # 清理旧临时文件
                    for p in [tmp_norm, tmp_mask]:
                        if os.path.exists(p):
                            os.remove(p)

                    # wahet 命令：使用绝对路径，在 BASE_DIR 下运行
                    cmd = [
                        WAHET,
                        "-i", img_path,
                        "-o", tmp_norm,
                        "-m", tmp_mask,
                    ]
                    try:
                        result = subprocess.run(
                            cmd, capture_output=True, text=True,
                            cwd=BASE_DIR, timeout=30
                        )
                        if result.returncode == 0 and os.path.exists(tmp_norm):
                            loaded = cv2.imread(tmp_norm, 0)
                            if loaded is not None:
                                norm_img = loaded
                                print(f"  wahet OK: norm {norm_img.shape}")
                            else:
                                print("  [warn] wahet 生成的 norm.bmp 无法读取，使用原图")
                        else:
                            print(f"  [warn] wahet 返回 {result.returncode}，使用原图")
                            if result.stderr.strip():
                                print(f"         {result.stderr.strip()[:120]}")
                    except subprocess.TimeoutExpired:
                        print("  [warn] wahet 超时，使用原图")
                    except Exception as e:
                        print(f"  [warn] wahet 调用失败: {e}，使用原图")
                    finally:
                        if not keep_temp:
                            for p in [tmp_norm, tmp_mask]:
                                if os.path.exists(p):
                                    os.remove(p)

                # ── 从原始眼部图提取圆（霍夫变换）───────────────────────
                # 注意：始终用原始眼部图 img_orig 做霍夫，不用 norm_img
                # 因为 norm_img 是展开后的矩形，已经没有圆结构
                pupil, iris_c = extract_circles_from_image(img_orig)
                print(f"  瞳孔: ({pupil[0]}, {pupil[1]}) r={pupil[2]}")
                print(f"  虹膜: ({iris_c[0]}, {iris_c[1]}) r={iris_c[2]}")

                # ── 生成掩码 ─────────────────────────────────────────────
                pup_mask = make_pupil_mask(H, W, pupil)
                iri_mask = make_iris_ring_mask(H, W, pupil, iris_c)

                # ── 验证掩码有效性 ───────────────────────────────────────
                if not is_valid_mask(pup_mask):
                    print(f"  [skip] 瞳孔掩码全黑，跳过")
                    fail += 1
                    continue
                if not is_valid_mask(iri_mask):
                    print(f"  [skip] 虹膜掩码全黑，跳过")
                    fail += 1
                    continue

                # ── 保存 ─────────────────────────────────────────────────
                # 训练图像：用原图（统一大小），如果 wahet 成功也可用 norm_img
                # 这里保存原图尺寸版本，与掩码尺寸一致
                out_img  = os.path.join(TRAIN_DATA_DIR, "images", base_name + ".png")
                out_pup  = os.path.join(TRAIN_DATA_DIR, "pupil",  base_name + ".png")
                out_iri  = os.path.join(TRAIN_DATA_DIR, "iris",   base_name + ".png")

                cv2.imwrite(out_img,  img_orig)
                cv2.imwrite(out_pup,  pup_mask)
                cv2.imwrite(out_iri,  iri_mask)

                success += 1
                print(f"  [OK] saved -> {base_name}")

    print(f"\n========================================")
    print(f"完成。成功: {success}  失败/跳过: {fail}")
    print(f"训练数据保存在: {TRAIN_DATA_DIR}")
    print(f"========================================")
    return success


if __name__ == "__main__":
    generate_training_data(use_wahet=True, keep_temp=False)