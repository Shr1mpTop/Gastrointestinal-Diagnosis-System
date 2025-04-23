# CAM图生成测试

import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from MedMamba import VSSM
from grad_cam import GradCAM
from MedMamba import SS_Conv_SSM

# --- 配置 ---
IMAGE_PATH    = "input.jpg"
GRAY_OUT      = "cam_gray.jpg"
OVERLAY_OUT   = "cam_overlay.jpg"
DEVICE        = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# --- 1. 加载原图 & 预处理 ---
orig_bgr = cv2.imread(IMAGE_PATH)
if orig_bgr is None:
    raise FileNotFoundError(f"找不到图像：{IMAGE_PATH}")
h0, w0 = orig_bgr.shape[:2]
orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
inp = transform(orig_rgb).unsqueeze(0).to(DEVICE)  # [1,3,224,224]

# --- 2. 初始化模型 & 定位目标层 ---
model = VSSM(num_classes=8).to(DEVICE)
model.eval()

# 定位最后一个 SS_Conv_SSM 模块中的 1×1 卷积  
last_block: SS_Conv_SSM = model.layers[-1].blocks[-1]  
# conv33conv33conv11 的第 7 层正好是最后一层 1×1 conv
target_layer = last_block.conv33conv33conv11[7]  

# --- 3. 初始化 GradCAM ---
gradcam = GradCAM(model, target_layer)

# --- 4. 生成灰度 CAM (0–255) ---
# 以类别 0 为例，也可以改为其他索引
cam_gray = gradcam.forward(inp, class_idx=0)  
# cam_gray: numpy (224×224)，uint8

# 保存灰度图
cv2.imwrite(GRAY_OUT, cam_gray)

# --- 5. 将灰度 CAM resize 回原图大小并叠加 ---
cam_resized = cv2.resize(cam_gray, (w0, h0), interpolation=cv2.INTER_LINEAR)
cam_color   = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
overlay     = cv2.addWeighted(orig_bgr, 0.5, cam_color, 0.5, 0)

# 保存叠加图
cv2.imwrite(OVERLAY_OUT, overlay)

print(f"✅ 灰度热力图已保存为 {GRAY_OUT}")
print(f"✅ 叠加热力图已保存为 {OVERLAY_OUT}")
