import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()  # 确保模型处于评估模式
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册前向/后向钩子（支持卷积层）
        def forward_hook(module, input, output):
            self.activations = output  # [B, C, H, W]
        def backward_hook(module, input_grad, output_grad):
            self.gradients = output_grad[0]  # [B, C, H, W]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def forward(self, input_tensor, class_idx):
        input_tensor = input_tensor.requires_grad_()  # 启用梯度计算
        output = self.model(input_tensor)
        score = output[0, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)  # 反向传播
        
        # 全局平均池化梯度
        grads = self.gradients
        activations = self.activations
        weights = torch.mean(grads, dim=(2, 3))  # [B, C]
        
        # 计算热力图
        cam = torch.zeros_like(activations[0], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam[i] = activations[0, i] * w
        cam = torch.sum(cam, dim=0)
        cam = F.relu(cam)  # 保留正相关特征
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # 归一化
        
        # 转换为图像格式（BGR顺序，适合OpenCV）
        cam = cv2.resize(cam.detach().cpu().numpy(), (input_tensor.shape[3], input_tensor.shape[2]))  # 先使用 detach() 方法，再使用 cpu() 方法将张量复制到 CPU 上
        cam = np.uint8(255 * cam)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # 生成热力图
        return cam

    