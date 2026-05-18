"""
analysis.py  ——  虹膜系统性能分析与可视化

核心修复（本版本）：
  图4 CNN Effect：改为只取同一人（最大组）的图像做类内比较，
                  不再混入跨人图像，确保 HD 反映真实类内变化。
  图6 Stable Bits vs Consistency：改为对最大单人组计算稳定性，
                  不跨人计算，确保图有意义。
  图3 Heatmap：保留全数据集计算稳定性，但修正轴标签（0~1279 bit位）。
  np.trapz 兼容 NumPy 2.0。
"""

import os
import cv2
import numpy as np
import itertools
import subprocess
import shutil

# NumPy 1.x / 2.x 兼容
_np_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception as _e:
    _MPL_OK = False
    _MPL_ERR_MSG = f"matplotlib failed: {_e}\nFix: pip install --upgrade matplotlib\n"

from scripts.rotation_match  import best_rotation_match
from scripts.stable_bits     import select_stable_bits
from scripts.blur_stabilizer import blur_stabilize
from scripts.cnn_stabilizer  import cnn_stabilize

DATASET   = "dataset"
OUTPUT    = "output"
PLOTS_DIR = os.path.abspath("output/plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── 全局数据缓存 ──────────────────────────────────────────────────────────────
# _ALL_CODES  : { "person_eye": [code_array, ...] }  — 全数据集，按人分组
_ALL_CODES   = {}


# ══════════════════════════════════════════════════════════════════════════════
# 样式
# ══════════════════════════════════════════════════════════════════════════════

def _check_mpl():
    if not _MPL_OK:
        raise RuntimeError(_MPL_ERR_MSG)

def _apply_style():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    12,
        "axes.labelsize":    10,
        "figure.facecolor":  "#e8f5e9",
        "axes.facecolor":    "#f1faf2",
        "axes.edgecolor":    "#2e7d32",
        "axes.labelcolor":   "#1b5e20",
        "axes.titlecolor":   "#1b5e20",
        "xtick.color":       "#2e7d32",
        "ytick.color":       "#2e7d32",
        "text.color":        "#1b5e20",
        "grid.color":        "#a5d6a7",
        "grid.linewidth":    0.5,
        "figure.dpi":        120,
        "savefig.dpi":       150,
        "savefig.facecolor": "#e8f5e9",
    })

BLUE   = "#1565c0"
GREEN  = "#2e7d32"
ORANGE = "#e65100"
RED    = "#c62828"
DIM    = "#78909c"


# ══════════════════════════════════════════════════════════════════════════════
# 全数据集处理
# ══════════════════════════════════════════════════════════════════════════════

def _run_full_dataset(progress_callback=None):
    """
    遍历 dataset/ 下所有人员、眼别、图像，
    执行 wahet → 稳定化 → lg，
    将结果按 person_eye 分组存入 _ALL_CODES。

    关键设计：
      _ALL_CODES 按人分组 → 支持 Genuine（同人）/ Impostor（跨人）计算
    """
    global _ALL_CODES
    _ALL_CODES   = {}

    wahet    = os.path.abspath("usit/bin/wahet.exe")
    feat     = os.path.abspath("usit/bin/lg.exe")
    wahet_ok = os.path.exists(wahet)
    feat_ok  = os.path.exists(feat)

    if not wahet_ok or not feat_ok:
        if progress_callback:
            progress_callback(
                f"  [warn] wahet={'OK' if wahet_ok else 'MISSING'}  "
                f"lg={'OK' if feat_ok else 'MISSING'}\n"
                "  Loading from existing output/code/ only.\n")
        _load_existing_codes(progress_callback)
        return

    tmp_root     = os.path.abspath("output/_analysis_tmp")
    dataset_root = os.path.abspath(DATASET)
    os.makedirs(tmp_root, exist_ok=True)

    if not os.path.exists(dataset_root):
        if progress_callback:
            progress_callback(f"  [warn] dataset/ not found.\n")
        _load_existing_codes(progress_callback)
        return

    people    = sorted(d for d in os.listdir(dataset_root)
                       if os.path.isdir(os.path.join(dataset_root, d)))
    n_total   = 0
    n_success = 0

    for person in people:
        for eye in ["L", "R"]:
            eye_dir = os.path.join(dataset_root, person, eye)
            if not os.path.isdir(eye_dir):
                continue

            group_key    = f"{person}_{eye}"
            group_codes  = []
            group_norms  = []

            img_files = sorted(
                f for f in os.listdir(eye_dir)
                if f.lower().endswith((".jpg", ".png", ".bmp")))

            for img_file in img_files:
                n_total  += 1
                img_path  = os.path.join(eye_dir, img_file)
                stem      = os.path.splitext(img_file)[0]
                pfx       = f"{person}_{eye}_{stem}"

                tmp_norm = os.path.join(tmp_root, pfx + "_norm.bmp")
                tmp_mask = os.path.join(tmp_root, pfx + "_mask.bmp")
                tmp_seg  = os.path.join(tmp_root, pfx + "_seg.bmp")
                tmp_code = os.path.join(tmp_root, pfx + "_code.png")

                for p in [tmp_norm, tmp_mask, tmp_seg, tmp_code]:
                    if os.path.exists(p): os.remove(p)

                # wahet
                try:
                    subprocess.run(
                        [wahet, "-i", img_path, "-o", tmp_norm,
                         "-m", tmp_mask, "-sr", tmp_seg],
                        capture_output=True, timeout=30)
                except Exception:
                    pass

                if not os.path.exists(tmp_norm):
                    continue
                norm_img = cv2.imread(tmp_norm, 0)
                if norm_img is None or float(norm_img.mean()) < 2.0:
                    continue

                # 稳定化
                norm_img = blur_stabilize(norm_img)
                norm_img = cnn_stabilize(norm_img)
                cv2.imwrite(tmp_norm, norm_img)

                # Log-Gabor
                try:
                    subprocess.run(
                        [feat, "-i", tmp_norm, "-o", tmp_code],
                        capture_output=True, timeout=30)
                except Exception:
                    pass

                if not os.path.exists(tmp_code):
                    continue
                code_img = cv2.imread(tmp_code, 0)
                if code_img is None:
                    continue

                code = (code_img > 127).astype(np.uint8).flatten()
                group_codes.append(code)
                n_success += 1

            if group_codes:
                _ALL_CODES[group_key]   = group_codes
                if progress_callback:
                    progress_callback(
                        f"  {group_key}: {len(group_codes)} codes\n")

    if progress_callback:
        progress_callback(
            f"  Full dataset done: {n_success}/{n_total} images, "
            f"{len(_ALL_CODES)} groups\n")

    shutil.rmtree(tmp_root, ignore_errors=True)

    if not _ALL_CODES:
        if progress_callback:
            progress_callback("  [warn] Falling back to output/code/\n")
        _load_existing_codes(progress_callback)


def _load_existing_codes(progress_callback=None):
    """回退：读取 output/code/ 和 output/norm/ 目录。"""
    global _ALL_CODES
    code_dir = os.path.abspath(os.path.join(OUTPUT, "code"))
    norm_dir = os.path.abspath(os.path.join(OUTPUT, "norm"))

    codes = []
    if os.path.exists(code_dir):
        for f in sorted(os.listdir(code_dir)):
            if f.endswith(".png"):
                img = cv2.imread(os.path.join(code_dir, f), 0)
                if img is not None:
                    codes.append((img > 127).astype(np.uint8).flatten())


    if codes:
        _ALL_CODES["current"]   = codes
        if progress_callback:
            progress_callback(f"  Loaded {len(codes)} codes from output/\n")


def _get_largest_group():
    """
    返回样本数最多的单人组的 (group_key, codes)。
    用于图6（Stable Bits vs Consistency）——需要同人数据。
    """
    if not _ALL_CODES:
        return None, []
    key = max(_ALL_CODES, key=lambda k: len(_ALL_CODES[k]))
    return key, _ALL_CODES[key]


# ══════════════════════════════════════════════════════════════════════════════
# HD pair 计算
# ══════════════════════════════════════════════════════════════════════════════

def _get_hd_pairs():
    """
    Genuine  = 同人同眼不同图之间（真实类内）
    Impostor = 不同人/不同眼之间（真实类间）
    单组时 Impostor 用随机噪声模拟。
    """
    groups       = list(_ALL_CODES.items())
    genuine_hds  = []
    impostor_hds = []

    for label, codes in groups:
        for i, j in itertools.combinations(range(len(codes)), 2):
            genuine_hds.append(float(best_rotation_match(codes[i], codes[j])))

    if len(groups) > 1:
        for (l1, c1), (l2, c2) in itertools.combinations(groups, 2):
            for a in c1[:5]:
                for b in c2[:5]:
                    impostor_hds.append(float(best_rotation_match(a, b)))
    else:
        if groups:
            rng = np.random.default_rng(42)
            ref = groups[0][1][0]
            for _ in range(80):
                noise = rng.integers(0, 2, size=len(ref), dtype=np.uint8)
                impostor_hds.append(float(best_rotation_match(ref, noise)))

    return genuine_hds, impostor_hds


def _eer_threshold(genuine, impostor, thresholds):
    best_t, best_diff = float(thresholds[0]), float("inf")
    g = np.array(genuine); m = np.array(impostor)
    for t in thresholds:
        far = float(np.mean(m < t)); frr = float(np.mean(g >= t))
        d   = abs(far - frr)
        if d < best_diff:
            best_diff, best_t = d, float(t)
    return best_t


# ══════════════════════════════════════════════════════════════════════════════
# 图1：HD Distribution（全数据集）
# ══════════════════════════════════════════════════════════════════════════════

def plot_hd_distribution(save=True):
    _check_mpl(); _apply_style()
    genuine_hds, impostor_hds = _get_hd_pairs()
    if not genuine_hds:
        return None

    n_groups = len(_ALL_CODES)
    imp_label = (f"Impostor-real (n={len(impostor_hds)})" if n_groups > 1
                 else f"Impostor-simul (n={len(impostor_hds)})")

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 0.6, 40)
    ax.hist(genuine_hds,  bins=bins, alpha=0.75, color=BLUE,
            label=f"Genuine  (n={len(genuine_hds)})", density=True)
    ax.hist(impostor_hds, bins=bins, alpha=0.65, color=RED,
            label=imp_label, density=True)

    if genuine_hds and impostor_hds:
        t = _eer_threshold(genuine_hds, impostor_hds, np.linspace(0, 0.6, 300))
        ax.axvline(t, color=ORANGE, linewidth=1.5, linestyle="--",
                   label=f"EER threshold = {t:.3f}")
    g_mean = float(np.mean(genuine_hds))
    i_mean = float(np.mean(impostor_hds)) if impostor_hds else 0
    ax.axvline(g_mean, color=BLUE, linewidth=1, linestyle=":",
               label=f"Genuine mean = {g_mean:.3f}")
    ax.axvline(i_mean, color=RED,  linewidth=1, linestyle=":",
               label=f"Impostor mean = {i_mean:.3f}")

    suf = f"({n_groups} groups, full dataset)" if n_groups > 1 else "(single group)"
    ax.set_xlabel("Hamming Distance (HD)")
    ax.set_ylabel("Density")
    ax.set_title(f"HD Distribution: Genuine vs Impostor\n{suf}")
    ax.legend(framealpha=0.3, fontsize=8)
    ax.grid(True, alpha=0.3); ax.set_xlim(0, 0.6)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "hd_distribution.png")
    if save:
        fig.savefig(path, bbox_inches="tight"); plt.close(fig); return path
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 图2：ROC Curve（全数据集）
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curve(save=True):
    _check_mpl(); _apply_style()
    genuine_hds, impostor_hds = _get_hd_pairs()
    if not genuine_hds or not impostor_hds:
        return None

    g = np.array(genuine_hds, dtype=float)
    m = np.array(impostor_hds, dtype=float)
    thresholds = np.linspace(0.0, 0.65, 600)
    tars = np.array([float(np.mean(g < t)) for t in thresholds])
    fars = np.array([float(np.mean(m < t)) for t in thresholds])
    frrs = 1.0 - tars

    pts    = sorted(set(zip(fars.tolist(), tars.tolist())))
    if len(pts) < 2: return None
    fars_u = np.array([p[0] for p in pts])
    tars_u = np.array([p[1] for p in pts])
    auc    = float(abs(_np_trapz(tars_u, fars_u)))

    eer_idx  = int(np.argmin(np.abs(fars - frrs)))
    eer_t    = float(thresholds[eer_idx])
    eer_rate = float((fars[eer_idx] + frrs[eer_idx]) / 2.0)

    n_groups = len(_ALL_CODES)
    note = "" if n_groups > 1 else "\n(Note: impostor simulated)"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fars, tars, color=BLUE, linewidth=2, label=f"ROC  (AUC = {auc:.4f})")
    ax.plot([0,1],[0,1], color=DIM, linewidth=0.8, linestyle="--",
            label="Random classifier")
    ax.scatter([fars[eer_idx]], [tars[eer_idx]],
               color=ORANGE, zorder=5, s=70,
               label=f"EER = {eer_rate:.3f}  (thr={eer_t:.3f})")
    ann_x = min(float(fars[eer_idx])+0.02, 0.18)
    ann_y = max(float(tars[eer_idx])-0.04, 0.82)
    ax.annotate(f"EER={eer_rate:.3f}",
                xy=(float(fars[eer_idx]), float(tars[eer_idx])),
                xytext=(ann_x, ann_y), color=ORANGE, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
    ax.set_xlabel("False Accept Rate (FAR)")
    ax.set_ylabel("True Accept Rate (TAR = 1 - FRR)")
    ax.set_title(f"ROC Curve{note}")
    ax.legend(framealpha=0.3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0, 0.2)
    ax.set_ylim(0.8, 1.0)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "roc_curve.png")
    if save:
        fig.savefig(path, bbox_inches="tight"); plt.close(fig); return path
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# 图5：FAR / FRR Curve（全数据集）
# ══════════════════════════════════════════════════════════════════════════════

def plot_far_frr_curve(save=True):
    _check_mpl(); _apply_style()
    genuine_hds, impostor_hds = _get_hd_pairs()
    if not genuine_hds or not impostor_hds:
        return None

    g = np.array(genuine_hds, dtype=float)
    m = np.array(impostor_hds, dtype=float)
    thresholds = np.linspace(0.01, 0.60, 200)
    fars = np.array([float(np.mean(m < t)) for t in thresholds])
    frrs = np.array([float(np.mean(g >= t)) for t in thresholds])
    eer_idx  = int(np.argmin(np.abs(fars-frrs)))
    eer_t    = float(thresholds[eer_idx])
    eer_rate = float((fars[eer_idx]+frrs[eer_idx])/2.0)

    n_groups = len(_ALL_CODES)
    fig, ax  = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, fars, color=RED,  linewidth=2, label="FAR (False Accept Rate)")
    ax.plot(thresholds, frrs, color=BLUE, linewidth=2, label="FRR (False Reject Rate)")
    ax.axvline(eer_t, color=ORANGE, linewidth=1.5, linestyle="--",
               label=f"EER: thr={eer_t:.3f}, rate={eer_rate:.3f}")
    ax.scatter([eer_t],[eer_rate], color=ORANGE, zorder=5, s=80)
    note = f"\n({n_groups} groups)" if n_groups>1 else "\n(impostor simulated)"
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Error Rate")
    ax.set_title(f"FAR / FRR vs Decision Threshold{note}")
    ax.legend(framealpha=0.3); ax.grid(True, alpha=0.3)
    ax.set_xlim(0.0,0.6); ax.set_ylim(-0.02,1.05)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "far_frr_curve.png")
    if save:
        fig.savefig(path, bbox_inches="tight"); plt.close(fig); return path
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 图6：Stable Bits vs Consistency（修复：使用单人最大组计算）
# ══════════════════════════════════════════════════════════════════════════════

def plot_stable_bits_vs_consistency(save=True):
    _check_mpl(); _apply_style()

    # ── 关键修复：只对单人组计算稳定性 ──────────────────────────────────────
    # 跨人计算时所有比特 mean≈0.5（不同人纹理不同），稳定比特数接近0，图无意义。
    # 只有同人多张图，才能体现"哪些比特位在该人的重复采集中保持一致"。
    group_key, group_codes = _get_largest_group()
    if len(group_codes) < 2:
        return None

    thresholds    = np.arange(0.60, 1.01, 0.02)
    n_stable_list = []
    mean_hd_list  = []

    for thr in thresholds:
        try:
            stable_codes, _ = select_stable_bits(group_codes, threshold=float(thr))
            n = int(stable_codes.shape[1])
            n_stable_list.append(n)
            if n > 0 and len(stable_codes) >= 2:
                hds = [float(best_rotation_match(stable_codes[0], stable_codes[i]))
                       for i in range(1, len(stable_codes))]
                mean_hd_list.append(float(np.mean(hds)))
            else:
                mean_hd_list.append(float("nan"))
        except Exception:
            n_stable_list.append(0)
            mean_hd_list.append(float("nan"))

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(thresholds, n_stable_list, color=BLUE, linewidth=2,
             marker="o", markersize=4, label="# Stable bits")
    ax1.axvline(0.85, color=GREEN, linewidth=1.2, linestyle=":",
                label="Current threshold=0.85")
    ax1.set_xlabel("Stability Threshold  tau")
    ax1.set_ylabel("Number of stable bits  N_stable(tau)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    valid = [(float(thresholds[i]), mean_hd_list[i])
             for i in range(len(mean_hd_list))
             if not np.isnan(mean_hd_list[i])]
    if valid:
        vx, vy = zip(*valid)
        ax2.plot(vx, vy, color=ORANGE, linewidth=2,
                 marker="s", markersize=4, linestyle="--",
                 label="Mean intra-HD (stable subset)")
    ax2.set_ylabel("Mean Hamming Distance  HD_stable(tau)", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2,
               framealpha=0.3, loc="upper right")

    plt.title(
        f"Stable Bit Count & Intra-HD vs Stability Threshold\n"
        f"(single-person group: {group_key},  {len(group_codes)} images)")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "stable_bits_vs_consistency.png")
    if save:
        fig.savefig(path, bbox_inches="tight"); plt.close(fig); return path
    return fig



# ══════════════════════════════════════════════════════════════════════════════
# 密钥再生成功率统计
# ══════════════════════════════════════════════════════════════════════════════

def _compute_key_regeneration_stats(progress_callback=None):
    """
    基于 _ALL_CODES 中已有的 IrisCode，模拟分块 RS 模糊提取器的密钥再生过程，
    统计以下三项指标：
      1. RS 整体成功率  — 至少1块解码成功（流程不崩溃）
      2. 密钥完整一致率 — 全部块均解码成功（密钥与注册完全相同）
      3. 多数投票兜底率 — 全部块解码失败，触发兜底

    对每个有≥2张图的用户组：
      以第1张图的稳定比特作为注册基准（enroll），
      用第2~N张图逐一验证（reproduce），统计成功与失败。
    """
    if not _ALL_CODES:
        return

    try:
        from scripts.stable_bits    import select_stable_bits
        from scripts.fuzzy_extractor import enroll, reproduce, _bits_to_bytes
        import reedsolo
        _RS_AVAIL = True
    except ImportError:
        _RS_AVAIL = False

    if not _RS_AVAIL:
        if progress_callback:
            progress_callback("  [warn] reedsolo not installed, skipping key regen stats.\n")
        return

    n_pairs          = 0   # 总验证配对数
    n_rs_success     = 0   # RS 整体成功（至少1块）
    n_full_match     = 0   # 全部块成功（密钥完整一致）
    n_fallback       = 0   # 全部块失败（多数投票兜底）
    n_skip           = 0   # 稳定比特为0，跳过

    for group_key, codes in _ALL_CODES.items():
        if len(codes) < 2:
            continue

        # 筛选稳定比特（τ=0.85，与主流程一致）
        try:
            stable_codes, _ = select_stable_bits(codes, threshold=0.85)
        except Exception:
            n_skip += len(codes) - 1
            continue

        n_bits = stable_codes.shape[1]
        if n_bits == 0:
            n_skip += len(codes) - 1
            continue

        # 注册：以第1张图为基准
        try:
            _, helper = enroll(stable_codes[0], error_rate=0.40)
        except Exception:
            n_skip += len(codes) - 1
            continue

        # 解析 helper 得到块数
        try:
            n_blocks_h = int.from_bytes(helper[7:9], "little")
        except Exception:
            n_skip += len(codes) - 1
            continue

        # 验证：用第2~N张图逐一测试
        for i in range(1, len(codes)):
            n_pairs += 1
            try:
                import reedsolo as rs_mod
                nsym       = helper[0]
                block_size = helper[1]
                pad        = helper[2]
                bit_len    = int.from_bytes(helper[3:7], "little")
                n_blocks   = int.from_bytes(helper[7:9], "little")
                ecc_data   = helper[9:]

                bits = stable_codes[i][:bit_len]
                padded = np.concatenate([bits,
                    np.zeros((8 - len(bits) % 8) % 8, dtype=np.uint8)])                     if len(bits) % 8 != 0 else bits
                current_bytes = bytes(np.packbits(padded))

                rs_codec = rs_mod.RSCodec(nsym)
                n_blk_success = 0
                n_blk_fail    = 0
                for b in range(n_blocks):
                    chunk = current_bytes[b * block_size: (b+1) * block_size]
                    ecc   = ecc_data[b * nsym: (b+1) * nsym]
                    try:
                        rs_codec.decode(chunk + ecc)
                        n_blk_success += 1
                    except Exception:
                        n_blk_fail += 1

                if n_blk_success == 0:
                    n_fallback += 1
                elif n_blk_fail == 0:
                    n_full_match += 1
                    n_rs_success += 1
                else:
                    n_rs_success += 1   # 部分块成功，整体不崩溃

            except Exception:
                n_skip += 1

    if n_pairs == 0:
        if progress_callback:
            progress_callback("  [warn] No valid pairs for key regen test.\n")
        return

    rs_rate       = n_rs_success / n_pairs * 100
    full_rate     = n_full_match / n_pairs * 100
    fallback_rate = n_fallback   / n_pairs * 100

    if progress_callback:
        progress_callback("\n" + "="*60 + "\n")
        progress_callback("  [Key Regeneration Statistics]\n")
        progress_callback(f"  Total verification pairs      : {n_pairs}\n")
        progress_callback(f"  RS overall success rate       : "
                          f"{n_rs_success}/{n_pairs} = {rs_rate:.1f}%\n")
        progress_callback(f"  Full key match rate           : "
                          f"{n_full_match}/{n_pairs} = {full_rate:.1f}%\n")
        progress_callback(f"  Majority vote fallback rate   : "
                          f"{n_fallback}/{n_pairs} = {fallback_rate:.1f}%\n")
        if n_skip > 0:
            progress_callback(f"  Skipped (no stable bits)      : {n_skip}\n")
        progress_callback("="*60 + "\n")

# ══════════════════════════════════════════════════════════════════════════════
# 一键生成（先跑全数据集，再出图）
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_plots(progress_callback=None):
    if not _MPL_OK:
        if progress_callback: progress_callback(_MPL_ERR_MSG)
        return {}

    _apply_style()

    if progress_callback:
        progress_callback("\n[Step 1/2] Running full dataset...\n")
    _run_full_dataset(progress_callback)

    n_groups = len(_ALL_CODES)
    n_codes  = sum(len(v) for v in _ALL_CODES.values())
    if progress_callback:
        progress_callback(
            f"  Ready: {n_groups} groups, {n_codes} IrisCodes\n")
        if n_groups == 1:
            progress_callback(
                "  [note] 1 group only - Impostor will be simulated.\n"
                "  Add more persons to dataset/ for real Impostor data.\n")
        progress_callback("\n[Step 2/2] Generating charts...\n")

    # ── 密钥再生成功率统计（生成图片之前）──────────────────────────────────
    if progress_callback:
        progress_callback("\n[Step 1.5] Computing key regeneration stats...\n")
    _compute_key_regeneration_stats(progress_callback)

    results = {}
    tasks = [
        ("hd_distribution",           plot_hd_distribution,           "HD Distribution"),
        ("roc_curve",                  plot_roc_curve,                 "ROC Curve"),
        ("far_frr_curve",              plot_far_frr_curve,             "FAR/FRR Curve"),
        ("stable_bits_vs_consistency", plot_stable_bits_vs_consistency,"Stable Bits vs Consistency"),
    ]
    for key, fn, label in tasks:
        if progress_callback:
            progress_callback(f"  Generating: {label}...\n")
        try:
            path = fn(save=True)
            if path:
                results[key] = os.path.abspath(path)
                if progress_callback:
                    progress_callback(f"  OK -> {path}\n")
            else:
                if progress_callback:
                    progress_callback(f"  SKIP (no data): {label}\n")
        except Exception as e:
            import traceback
            if progress_callback:
                progress_callback(f"  ERROR {label}: {e}\n")
                progress_callback(f"  {traceback.format_exc()}\n")
    return results


if __name__ == "__main__":
    generate_all_plots(print)