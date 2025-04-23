# 注意力图生成测试

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from MedMamba import VSSM, SS_Conv_SSM, SparseAttention, SEBlock

# 1. 配置
IMAGE_PATH = "input.jpg"          # 你的输入图像路径
DEVICE     = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. 加载并预处理图像
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    # 根据训练时的 normalize 参数替换下面的 mean/std
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])
img = Image.open(IMAGE_PATH).convert("RGB")
inp = transform(img).unsqueeze(0).to(DEVICE)  # [1,3,224,224]

# 3. 构建模型并切换到 eval 模式
model = VSSM(num_classes=8).to(DEVICE)
model.eval()

# 4. 用于保存注意力
sparse_maps = []
se_weights  = []

# 5. 定义 hook：重算并保存 local sparse-attn
def save_sparse(module, inp, out):
    x = inp[0]  # [B,C,H,W]
    B, C, H, W = x.shape
    pad = module.window_size // 2

    q = module.q_proj(x).view(B, module.num_heads, module.head_dim, H, W)
    k = module.k_proj(x).view(B, module.num_heads, module.head_dim, H, W)
    k_pad = F.pad(k, (pad, pad, pad, pad))
    k_win = (k_pad
             .unfold(3, module.window_size, 1)
             .unfold(4, module.window_size, 1)
             .contiguous()
             .view(B, module.num_heads, module.head_dim, H, W, -1))
    attn = (q.unsqueeze(-1) * k_win).sum(dim=2) * module.scale
    attn = attn.softmax(dim=-1)
    sparse_maps.append(attn[0, 0].mean(dim=-1).cpu())  # [H, W]

# 6. 定义 hook：重算并保存 SEBlock 的通道注意力
def save_se(module, inp, out):
    x = inp[0]           # [B,C,H,W]
    b, c, h, w = x.shape
    y = module.avg_pool(x).view(b, c)   # [B, C]
    wts = module.fc(y).cpu()            # [B, C]
    se_weights.append(wts[0])           # [C]

# 7. 注册 hooks 到所有 SS_Conv_SSM 实例
for m in model.modules():
    if isinstance(m, SS_Conv_SSM):
        # sparse-attention 分支
        sa = m.sparse_attention
        if isinstance(sa, SparseAttention):
            sa.register_forward_hook(save_sparse)
        # 左分支的 SEBlock
        for sub in m.modules():
            if isinstance(sub, SEBlock):
                sub.register_forward_hook(save_se)

# 8. 前向推理
with torch.no_grad():
    _ = model(inp)

# 9. 可视化并保存第一个 SS_Conv_SSM sparse-attn 热力图
attn0 = sparse_maps[0].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
attn0 = F.interpolate(attn0, size=(224,224), mode='bilinear', align_corners=False)[0,0]
plt.figure(figsize=(6,6))
plt.imshow(attn0, cmap='jet')
plt.axis('off')
plt.title('SS_Conv_SSM SparseAttention')
plt.savefig(os.path.join(OUTPUT_DIR, 'sparse_attn.png'), dpi=150)
plt.close()

# 10. 可视化并保存第一个 SEBlock 通道权重条形图
w = se_weights[0].numpy()  # [C]
plt.figure(figsize=(8,2))
plt.bar(range(len(w)), w)
plt.title('SS_Conv_SSM SEBlock Channel Weights')
plt.xlabel('Channel')
plt.ylabel('Weight')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'se_weights.png'), dpi=150)
plt.close()

print(f"✅ Sparse-Attn heatmap saved to {OUTPUT_DIR}/sparse_attn.png")
print(f"✅ SE weights plot   saved to {OUTPUT_DIR}/se_weights.png")

