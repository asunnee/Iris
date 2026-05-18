"""
train.py  ——  训练轻量 U-Net 虹膜分割模型
放在项目根目录

依赖：pip install torch torchvision

用法：
  1. 先运行 prepare_training_data.py 生成训练数据
  2. python train.py
  3. 训练完成后权重保存到 models/iris_seg.pth
"""

import os
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    print("ERROR: PyTorch not installed. Run: pip install torch torchvision")
    exit(1)

TRAIN_DATA_DIR = os.path.join("dataset", "train_data")
MODEL_DIR      = "models"
MODEL_PATH     = os.path.join(MODEL_DIR, "iris_seg.pth")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 超参数 ─────────────────────────────────────────────────────────────────
IMG_H      = 256    # 训练时统一缩放尺寸
IMG_W      = 256
EPOCHS     =25
LR         = 1e-3
BATCH_SIZE = 4
VAL_SPLIT  = 0.15   # 15% 用于验证


# ══════════════════════════════════════════════════════════════════════════════
# 数据集
# ══════════════════════════════════════════════════════════════════════════════

class IrisSegDataset(Dataset):
    """
    读取 prepare_training_data.py 生成的训练数据。
    目录结构：
      train_data/images/  *.png  原始眼部图
      train_data/pupil/   *.png  瞳孔掩码
      train_data/iris/    *.png  虹膜环掩码
    """
    def __init__(self, data_root, file_list, img_size=(IMG_H, IMG_W)):
        self.img_dir   = os.path.join(data_root, "images")
        self.pup_dir   = os.path.join(data_root, "pupil")
        self.iri_dir   = os.path.join(data_root, "iris")
        self.files     = file_list
        self.img_size  = img_size   # (H, W)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name  = self.files[idx]
        img   = cv2.imread(os.path.join(self.img_dir, name), 0)
        pup   = cv2.imread(os.path.join(self.pup_dir, name), 0)
        iri   = cv2.imread(os.path.join(self.iri_dir, name), 0)

        # 统一缩放
        H, W = self.img_size
        img  = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        pup  = cv2.resize(pup, (W, H), interpolation=cv2.INTER_NEAREST)
        iri  = cv2.resize(iri, (W, H), interpolation=cv2.INTER_NEAREST)

        # 归一化 + 转 tensor
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)   # [1,H,W]
        pup_t = torch.from_numpy((pup > 127).astype(np.float32)).unsqueeze(0)   # [1,H,W]
        iri_t = torch.from_numpy((iri > 127).astype(np.float32)).unsqueeze(0)   # [1,H,W]

        mask_t = torch.cat([pup_t, iri_t], dim=0)   # [2,H,W]
        return img_t, mask_t


# ══════════════════════════════════════════════════════════════════════════════
# 轻量 U-Net
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class LightUNet(nn.Module):
    """
    输入：[B, 1, H, W]  灰度图
    输出：[B, 2, H, W]  logits（通道0=瞳孔，通道1=虹膜环）
    """
    def __init__(self, base=16):
        super().__init__()
        self.enc1 = ConvBlock(1,      base)
        self.enc2 = ConvBlock(base,   base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bot  = ConvBlock(base*4, base*8)
        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = ConvBlock(base*8, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base,   2, stride=2)
        self.dec1 = ConvBlock(base*2, base)
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


# ══════════════════════════════════════════════════════════════════════════════
# 损失函数：BCE + Dice
# ══════════════════════════════════════════════════════════════════════════════

def dice_loss(pred, target, eps=1e-6):
    pred   = torch.sigmoid(pred)
    inter  = (pred * target).sum(dim=(2, 3))
    union  = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice   = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def combined_loss(pred, target):
    bce  = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


# ══════════════════════════════════════════════════════════════════════════════
# IoU 评估
# ══════════════════════════════════════════════════════════════════════════════

def compute_iou(pred_logits, target, threshold=0.5):
    pred   = (torch.sigmoid(pred_logits) > threshold).float()
    inter  = (pred * target).sum(dim=(2, 3))
    union  = ((pred + target) > 0).float().sum(dim=(2, 3))
    iou    = (inter + 1e-6) / (union + 1e-6)
    return float(iou.mean())


# ══════════════════════════════════════════════════════════════════════════════
# 训练主函数
# ══════════════════════════════════════════════════════════════════════════════

def train():
    # ── 检查数据 ──────────────────────────────────────────────────────────
    img_dir = os.path.join(TRAIN_DATA_DIR, "images")
    if not os.path.exists(img_dir):
        print(f"ERROR: 训练数据不存在: {img_dir}")
        print("请先运行: python prepare_training_data.py")
        return

    all_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
    if len(all_files) < 4:
        print(f"ERROR: 训练数据太少 ({len(all_files)} 张)，至少需要 4 张。")
        print("请先运行: python prepare_training_data.py")
        return

    # 过滤掉掩码文件缺失的样本
    valid_files = []
    for f in all_files:
        pup_ok = os.path.exists(os.path.join(TRAIN_DATA_DIR, "pupil", f))
        iri_ok = os.path.exists(os.path.join(TRAIN_DATA_DIR, "iris",  f))
        if pup_ok and iri_ok:
            valid_files.append(f)
    print(f"有效训练样本: {len(valid_files)} / {len(all_files)}")

    # ── 划分训练/验证集 ────────────────────────────────────────────────────
    np.random.seed(42)
    np.random.shuffle(valid_files)
    n_val    = max(1, int(len(valid_files) * VAL_SPLIT))
    val_list = valid_files[:n_val]
    trn_list = valid_files[n_val:]
    print(f"训练: {len(trn_list)}  验证: {len(val_list)}")

    trn_set = IrisSegDataset(TRAIN_DATA_DIR, trn_list)
    val_set = IrisSegDataset(TRAIN_DATA_DIR, val_list)
    trn_ldr = DataLoader(trn_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=False)
    val_ldr = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0)

    # ── 模型 ──────────────────────────────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model     = LightUNet(base=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_val_iou = 0.0

    for epoch in range(1, EPOCHS + 1):
        # 训练
        model.train()
        trn_loss = 0.0
        for imgs, masks in trn_ldr:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = combined_loss(preds, masks)
            loss.backward()
            optimizer.step()
            trn_loss += loss.item()
        trn_loss /= max(len(trn_ldr), 1)

        # 验证
        model.eval()
        val_loss = 0.0
        val_iou  = 0.0
        with torch.no_grad():
            for imgs, masks in val_ldr:
                imgs, masks = imgs.to(device), masks.to(device)
                preds     = model(imgs)
                val_loss += combined_loss(preds, masks).item()
                val_iou  += compute_iou(preds, masks)
        val_loss /= max(len(val_ldr), 1)
        val_iou  /= max(len(val_ldr), 1)

        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={trn_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_IoU={val_iou:.4f}")

        # 保存最优模型
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  --> 模型已保存 (best IoU={best_val_iou:.4f})")

    print(f"\n训练完成。最优 IoU={best_val_iou:.4f}")
    print(f"模型权重: {MODEL_PATH}")


if __name__ == "__main__":
    train()