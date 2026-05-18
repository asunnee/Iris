"""
cnn_segmentor.py  ——  CNN 辅助虹膜分割模块
放在 scripts/ 目录

功能：
  1. 加载训练好的 U-Net 模型（models/iris_seg.pth）
  2. 对输入眼部图像预测瞳孔掩码 + 虹膜掩码
  3. 用 CNN 掩码约束霍夫变换搜索范围，精确定位边界圆
  4. 若模型不可用，自动退回纯霍夫变换

依赖：pip install torch torchvision
"""

import os
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

# ── 修复路径问题：使用 abspath 确保在任意工作目录下都能找到模型 ──────────────
# cnn_segmentor.py 在 scripts/ 下，上一级就是项目根目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # scripts/ 的绝对路径
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)                 # 项目根目录的绝对路径
MODEL_PATH  = os.path.join(_ROOT_DIR, "models", "iris_seg.pth")

IMG_SIZE = 256   # 与 train.py 中的 IMG_H / IMG_W 保持一致


# ══════════════════════════════════════════════════════════════════════════════
# U-Net 定义（与 train.py 完全一致，必须保持同步）
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH_OK:
    class _CB(nn.Module):
        """两层 Conv2d + BatchNorm + ReLU 卷积块。"""
        def __init__(self, ic, oc):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.net(x)

    class LightUNet(nn.Module):
        """
        轻量 U-Net（base=16 通道）
        输入：[B, 1, H, W]  灰度图
        输出：[B, 2, H, W]  logits（通道0=瞳孔，通道1=虹膜环）
        """
        def __init__(self, base=16):
            super().__init__()
            self.enc1 = _CB(1,      base)
            self.enc2 = _CB(base,   base*2)
            self.enc3 = _CB(base*2, base*4)
            self.pool = nn.MaxPool2d(2)
            self.bot  = _CB(base*4, base*8)
            self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
            self.dec3 = _CB(base*8, base*4)
            self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
            self.dec2 = _CB(base*4, base*2)
            self.up1  = nn.ConvTranspose2d(base*2, base,   2, stride=2)
            self.dec1 = _CB(base*2, base)
            self.out  = nn.Conv2d(base, 2, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b  = self.bot(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)


# ── 模型单例缓存（避免每次调用都重新加载） ────────────────────────────────────
_model_cache = None


def _load_model():
    """加载模型，成功返回模型对象，失败返回 None。"""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not _TORCH_OK:
        print("[cnn_segmentor] PyTorch 未安装，无法使用 CNN 模式")
        return None
    if not os.path.exists(MODEL_PATH):
        print(f"[cnn_segmentor] 模型文件不存在: {MODEL_PATH}")
        print(f"[cnn_segmentor] 请先运行: python prepare_training_data.py && python train.py")
        return None
    try:
        model = LightUNet(base=16)
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _model_cache = model
        print(f"[cnn_segmentor] 模型加载成功: {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"[cnn_segmentor] 模型加载失败: {e}")
        return None


def model_available():
    """外部检查接口：CNN 模型是否可用。"""
    return _load_model() is not None


# ══════════════════════════════════════════════════════════════════════════════
# CNN 推理：预测瞳孔和虹膜掩码
# ══════════════════════════════════════════════════════════════════════════════

def predict_masks(eye_img_gray):
    """
    输入：灰度眼部图（numpy uint8，任意尺寸）
    输出：(pupil_mask, iris_mask)
          - 二值图（uint8，0 或 255），与输入同尺寸
          - 若 CNN 不可用，返回 (None, None)
    """
    model = _load_model()
    if model is None:
        return None, None

    H_orig, W_orig = eye_img_gray.shape

    # 缩放到训练尺寸
    resized = cv2.resize(eye_img_gray, (IMG_SIZE, IMG_SIZE))

    # 转 tensor [1, 1, H, W]，归一化到 [0, 1]
    t = torch.from_numpy(
        resized.astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        probs = torch.sigmoid(model(t))   # [1, 2, H, W]

    pup_prob = probs[0, 0].numpy()   # 瞳孔概率图
    iri_prob = probs[0, 1].numpy()   # 虹膜环概率图

    # 二值化（阈值 0.5）
    pup_mask = (pup_prob > 0.5).astype(np.uint8) * 255
    iri_mask = (iri_prob > 0.5).astype(np.uint8) * 255

    # 还原到原始尺寸
    pup_mask = cv2.resize(pup_mask, (W_orig, H_orig),
                           interpolation=cv2.INTER_NEAREST)
    iri_mask = cv2.resize(iri_mask, (W_orig, H_orig),
                           interpolation=cv2.INTER_NEAREST)
    return pup_mask, iri_mask


# ══════════════════════════════════════════════════════════════════════════════
# 从掩码估算圆参数
# ══════════════════════════════════════════════════════════════════════════════

def _mask_to_circle(mask, fallback=None):
    """
    用最小外接圆估算掩码对应的圆心和半径。
    返回 (cx, cy, r) 或 fallback。
    """
    if mask is None:
        return fallback
    cnts, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return fallback
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 20:
        return fallback
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    return int(cx), int(cy), max(2, int(r))


# ══════════════════════════════════════════════════════════════════════════════
# 纯霍夫变换（不依赖 CNN，作为兜底）
# ══════════════════════════════════════════════════════════════════════════════

def _hough_only(eye_img_gray):
    """
    不使用 CNN，直接在全图上做霍夫圆检测。
    返回 (pupil, iris_c) 各为 (cx, cy, r)。
    """
    H, W    = eye_img_gray.shape
    blurred = cv2.GaussianBlur(eye_img_gray, (7, 7), 1.5)

    # 瞳孔：选最暗的小圆
    pupil   = None
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1,
        minDist=int(H * 0.1),
        param1=60, param2=20,
        minRadius=max(5,  int(H * 0.08)),
        maxRadius=max(10, int(H * 0.30))
    )
    if circles is not None:
        best_c, best_val = None, 255.0
        for c in circles[0]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            tmp = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(tmp, (cx, cy), r, 255, -1)
            mean_val = float(cv2.mean(eye_img_gray, mask=tmp)[0])
            if mean_val < best_val:
                best_val, best_c = mean_val, (cx, cy, r)
        pupil = best_c

    # 虹膜：选最靠近图像中心的大圆
    iris_c  = None
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1,
        minDist=int(H * 0.1),
        param1=50, param2=18,
        minRadius=max(10, int(H * 0.28)),
        maxRadius=max(20, int(H * 0.56))
    )
    if circles is not None:
        cx_img, cy_img = W // 2, H // 2
        best_c, best_d = None, float("inf")
        for c in circles[0]:
            cx, cy, r = int(c[0]), int(c[1]), int(c[2])
            d = (cx - cx_img)**2 + (cy - cy_img)**2
            if d < best_d:
                best_d, best_c = d, (cx, cy, r)
        iris_c = best_c

    # 兜底估算
    if pupil  is None: pupil  = (W//2, H//2, int(H * 0.14))
    if iris_c is None: iris_c = (W//2, H//2, int(H * 0.42))
    if pupil[2] >= iris_c[2]:
        pupil, iris_c = iris_c, pupil

    return pupil, iris_c


# ══════════════════════════════════════════════════════════════════════════════
# 核心接口：CNN 掩码约束的增强霍夫分割
# ══════════════════════════════════════════════════════════════════════════════

def enhanced_hough_segmentation(eye_img_gray):
    """
    CNN + 霍夫联合分割（CNN 为主，霍夫精修，纯霍夫兜底）。

    流程：
      1. CNN 预测瞳孔/虹膜概率掩码
      2. 从掩码估算圆心和半径的粗略范围
      3. 在约束范围内执行霍夫圆检测（精确定位）
      4. 全程三级兜底：CNN掩码圆 → 全图霍夫 → 固定估算值

    返回：
      {
        "pupil":  (cx, cy, r),
        "iris":   (cx, cy, r),
        "method": "cnn+hough" | "hough_only"
      }
    """
    H, W    = eye_img_gray.shape
    blurred = cv2.GaussianBlur(eye_img_gray, (7, 7), 1.5)

    # ── 第一步：CNN 预测掩码 ──────────────────────────────────────────────
    pup_mask, iri_mask = predict_masks(eye_img_gray)
    method = "cnn+hough" if pup_mask is not None else "hough_only"

    if pup_mask is None:
        # CNN 不可用，直接用纯霍夫
        pupil, iris_c = _hough_only(eye_img_gray)
        return {"pupil": pupil, "iris": iris_c, "method": "hough_only"}

    # ── 第二步：从掩码估算约束范围 ──────────────────────────────────────
    cnn_pupil = _mask_to_circle(pup_mask)
    cnn_iris  = _mask_to_circle(iri_mask)

    def hough_in_range(r_min, r_max, cx_hint=None, cy_hint=None, param2=20):
        """在指定半径范围内做霍夫检测，可选圆心 ROI 约束。"""
        r_min = max(3, int(r_min))
        r_max = max(r_min + 2, int(r_max))

        if cx_hint is not None:
            margin = int(r_max * 1.3) + 20
            x1 = max(0, cx_hint - margin)
            y1 = max(0, cy_hint - margin)
            x2 = min(W, cx_hint + margin)
            y2 = min(H, cy_hint + margin)
            roi    = blurred[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            roi    = blurred
            offset = (0, 0)

        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return None

        circles = cv2.HoughCircles(
            roi, cv2.HOUGH_GRADIENT, dp=1,
            minDist=max(r_min, 8),
            param1=50, param2=param2,
            minRadius=r_min, maxRadius=r_max
        )
        if circles is None:
            return None
        c = circles[0][0]
        return (int(c[0]) + offset[0],
                int(c[1]) + offset[1],
                int(c[2]))

    # ── 第三步：瞳孔精确定位（三级兜底）──────────────────────────────────
    pupil = None

    # 级别1：CNN 掩码约束的局部霍夫
    if cnn_pupil is not None:
        cx, cy, r = cnn_pupil
        pupil = hough_in_range(
            max(3, r - 12), r + 12,
            cx, cy, param2=15
        )

    # 级别2：全图霍夫（放宽半径范围）
    if pupil is None:
        pupil = hough_in_range(
            max(5, int(H * 0.07)), int(H * 0.32),
            param2=18
        )

    # 级别3：直接用 CNN 掩码圆（不做霍夫）
    if pupil is None and cnn_pupil is not None:
        pupil = cnn_pupil

    # 级别4：固定估算兜底
    if pupil is None:
        pupil = (W // 2, H // 2, int(H * 0.14))

    # ── 第四步：虹膜精确定位（三级兜底）──────────────────────────────────
    iris_c = None

    if cnn_iris is not None:
        cx, cy, r = cnn_iris
        iris_c = hough_in_range(
            max(10, r - 18), r + 18,
            cx, cy, param2=20
        )

    if iris_c is None:
        iris_c = hough_in_range(
            int(H * 0.27), int(H * 0.57),
            param2=18
        )

    if iris_c is None and cnn_iris is not None:
        iris_c = cnn_iris

    if iris_c is None:
        iris_c = (W // 2, H // 2, int(H * 0.42))

    # ── 保证瞳孔半径 < 虹膜半径 ──────────────────────────────────────────
    if pupil[2] >= iris_c[2]:
        pupil, iris_c = iris_c, pupil

    return {"pupil": pupil, "iris": iris_c, "method": method}


# ══════════════════════════════════════════════════════════════════════════════
# 可视化（调试用）
# ══════════════════════════════════════════════════════════════════════════════

def visualize_segmentation(eye_img_gray, save_path=None):
    """
    在原图上叠加 CNN 掩码（半透明）和霍夫圆，返回 BGR 可视化图像。
    """
    result = enhanced_hough_segmentation(eye_img_gray)
    vis    = cv2.cvtColor(eye_img_gray, cv2.COLOR_GRAY2BGR)

    pup_mask, iri_mask = predict_masks(eye_img_gray)
    if pup_mask is not None and iri_mask is not None:
        overlay = vis.copy()
        overlay[pup_mask > 0] = (200, 80,  80)    # 蓝色调 = 瞳孔区域
        overlay[iri_mask > 0] = (80,  180, 80)    # 绿色调 = 虹膜区域
        vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0)

    # 绘制检测到的圆
    cx, cy, r = result["pupil"]
    cv2.circle(vis, (cx, cy), r, (0, 0, 220), 2)   # 红色 = 瞳孔

    cx, cy, r = result["iris"]
    cv2.circle(vis, (cx, cy), r, (0, 200, 0), 2)   # 绿色 = 虹膜

    cv2.putText(vis, result["method"], (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1)

    if save_path:
        cv2.imwrite(save_path, vis)
    return vis