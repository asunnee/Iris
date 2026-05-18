# 修正后的 debug_codes
import os
import cv2
import numpy as np

code_dir = "output/code"
print(f"正在检查 IrisCode 图像路径: {code_dir}...\n")

if not os.path.exists(code_dir):
    print("错误：路径不存在。")
else:
    for file in sorted(os.listdir(code_dir)):
        if file.endswith(".png"):
            path = os.path.join(code_dir, file)
            img = cv2.imread(path, 0)
            if img is None: continue

            # 转换为二值，计算 1 的占比
            binary = (img > 127).astype(np.uint8)
            mean_val = np.mean(binary)
            # 如果 mean 接近 0 或 1，说明特征提取失败，是全黑或全白
            status = "正常" if 0.2 < mean_val < 0.8 else "警告：可能无效"
            print(f"文件: {file:<20} | 均值(1占比): {mean_val:.4f} | 状态: {status}")