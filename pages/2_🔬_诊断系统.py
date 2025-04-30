import streamlit as st
import torch
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO
import base64
from torchvision import transforms
import cv2
import torch.nn.functional as F
from grad_cam import GradCAM
from MedMamba import SS_Conv_SSM, SparseAttention, SEBlock
from fpdf import FPDF
import tempfile
from matplotlib.font_manager import FontProperties
import uuid

# --- Configuration ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Page Title ---
st.title("AI病理诊断系统")

# --- Check if Model and Explainer are Loaded ---
if 'model' not in st.session_state or not st.session_state.get('model_loaded', False):
    st.error("模型没有完成加载，请检查日志以及系统首页的体统状态")
    st.stop() 

if 'explainer' not in st.session_state or not st.session_state.get('explainer_loaded', False):
    st.warning("大语言模型没有完成加载，诊断依然有效但是不会有病理解释")

def tensor_to_image(tensor):
    tensor = tensor.squeeze().cpu()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean  # Denormalize
    tensor = tensor.clamp(0, 1)
    return tensor.permute(1, 2, 0).numpy()  # [H, W, 3], float32 [0,1]

def create_overlay(input_img, heatmap, alpha=0.5):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)  # Normalize to [0,1]
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0
    overlay = input_img * (1 - alpha) + heatmap_color * alpha
    return np.clip(overlay, 0, 1)

# --- Preprocessing Pipeline ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Load Class Labels ---
try:
    with open('class_indices.json', 'r') as f:
        class_indict = json.load(f)
    labels = [v for _, v in class_indict.items()]
except FileNotFoundError:
    st.error("发生错误: '未找到 “class_indices.json” 文件。无法将预测结果映射到标签上。")
    st.stop()
except Exception as e:
    st.error(f"在加载标签哈希表时发生错误: {str(e)}")
    st.stop()

# --- Sidebar for Diagnosis Page ---
with st.sidebar:
    st.header("使用说明")
    st.markdown("""
    1. 上传内窥镜图像(JPEG/PNG格式)
    2. 系统将自动进行病理分析
    3. 查看诊断结果及置信度评分
    4. 展开下方区域查看详细分析和AI生成报告
    """)
    st.divider()
    if st.button("🛠️ 系统健康检查"):
        with st.status("执行检查中", expanded=True) as status:
            check_results = []
            try:
                import torch, PIL, numpy
                check_results.append(("核心库", "✅ 完成", f"Torch {torch.__version__}"))
            except ImportError as e:
                check_results.append(("核心库", "❌ 失败", f"Missing: {str(e)}"))
            if 'model' in st.session_state and st.session_state.get('model_loaded', False):
                 check_results.append(("病理模型", "✅ 加载完成", f"设备: {device}"))
            else:
                 check_results.append(("病理模型", "❌ 未能加载", "检查首页状态"))
            if 'explainer' in st.session_state and st.session_state.get('explainer_loaded', False):
                 check_results.append(("AI诊断", "✅ 初始化", "完成"))
            else:
                 check_results.append(("AI诊断", "❌ 初始化失败", "检查主页状态 / API 密钥"))
            status.update(label="Diagnostics Complete", state="complete")
            st.table(pd.DataFrame(check_results, columns=["组件", "状态", "详情"]))

# --- Main Interface ---
uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
col1, col2 = st.columns(2)

if uploaded_file is not None:
    try:
        # Image Preprocessing
        img = Image.open(uploaded_file).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)

        # Set up attention capture
        sparse_maps = []
        se_weights = []
        se_attn_maps = []  # 新增：存储 SE 空间注意力图

        # Hook functions for attention
        def save_sparse(module, inp, out):
            x = inp[0]  # [B, C, H, W]
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

        def save_se(module, inp, out):
            x = inp[0]  # [B, C, H, W]
            b, c, h, w = x.shape
            y = module.avg_pool(x).view(b, c)  # [B, C]
            wts = module.fc(y).cpu()  # [B, C]
            se_weights.append(wts[0])  # [C]
            se_attn_maps.append(out.mean(dim=1)[0].cpu())  # [H, W]

        # Register hooks
        model = st.session_state.model
        handles = []
        for m in model.modules():
            if isinstance(m, SparseAttention):
                handles.append(m.register_forward_hook(save_sparse))
            elif isinstance(m, SEBlock):
                handles.append(m.register_forward_hook(save_se))

        # Model Inference
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0] * 100

        # Remove hooks
        for h in handles:
            h.remove()

        # Display Input Image
        with col1:
            st.image(img, caption="输入图像", use_container_width=True)

        # Display Diagnosis Result
        with col2:
            pred_idx = torch.argmax(probs).item()
            pred_label = labels[pred_idx]
            pred_conf = probs[pred_idx].item()
            st.subheader(f"诊断结果：{labels[pred_idx]}")
            st.metric(label="置信度", value=f"{probs[pred_idx]:.1f}%")

        # Enhanced Results Display
        with st.expander("📊 详细诊断分析", expanded=True):
            tab1, tab2 = st.tabs(["置信度分布", "临床建议"])
            y_pos = np.arange(len(labels))
            with tab1:
                font = FontProperties(fname='SimHei.ttf')
                fig_conf, ax_conf = plt.subplots(figsize=(10, 5))
                ax_conf.barh(y_pos, probs.cpu().numpy(), align='center')
                ax_conf.set_yticks(y_pos)
                ax_conf.set_yticklabels(labels, fontproperties=font)
                ax_conf.invert_yaxis()
                ax_conf.set_xlabel("可信度 (%)", fontproperties=font)
                ax_conf.set_title("不同类别的置信度", fontproperties=font)
                st.pyplot(fig_conf)

            with tab2:
                st.markdown("""
                **临床处理建议（示例指南）**
                * **高置信度 (>85%):** 建议根据发现立即采取行动或会诊
                * **中置信度 (50%-85%):** 建议进一步检查或专科会诊
                * **低置信度 (<50%):** 建议观察随访
                * *免责声明：本指南仅为示例，不构成医疗建议*
                """)

        # Get 224x224 input image for overlays
        input_img = tensor_to_image(input_tensor)

        # Generate CAM
        try:
            model.eval()
            last_block = model.layers[-1].blocks[-1]
            target_layer = last_block.conv33conv33conv11[7]
            gradcam = GradCAM(model, target_layer)
            cam_gray = gradcam.forward(input_tensor, class_idx=pred_idx)
            cam_overlay = create_overlay(input_img, cam_gray)
        except Exception as e:
            st.error(f"生成CAM失败: {str(e)}")
            cam_overlay = None

        # Generate sparse attention map
        attn_overlay = None
        if sparse_maps:
            try:
                attn0 = sparse_maps[-1]  # 使用最后一个注意力图
                attn0 = F.interpolate(attn0.unsqueeze(0).unsqueeze(0), size=(224, 224),
                                      mode='bilinear', align_corners=False)[0, 0].numpy()
                attn_overlay = create_overlay(input_img, attn0)
            except Exception as e:
                st.error(f"注意力图生成失败: {str(e)}")

        # Generate SE attention map
        se_attn_map = se_attn_maps[-1] if se_attn_maps else None
        if se_attn_map is not None:
            try:
                se_attn_map = F.interpolate(se_attn_map.unsqueeze(0).unsqueeze(0), size=(224, 224),
                                            mode='bilinear', align_corners=False)[0, 0]
                se_attn_heatmap = (se_attn_map - se_attn_map.min()) / (se_attn_map.max() - se_attn_map.min() + 1e-8)
                se_attn_overlay = create_overlay(input_img, se_attn_heatmap.numpy())
            except Exception as e:
                st.error(f"SE注意力图生成失败: {str(e)}")
                se_attn_overlay = None
        else:
            se_attn_overlay = None

        # Interpretability Section
        with st.expander("🔍 模型可解释性分析", expanded=False):
            cols = st.columns(3)
            if cam_overlay is not None:
                with cols[0]:
                    st.subheader("类别激活图 (CAM)")
                    st.image(cam_overlay, caption="CAM可视化叠加效果（处理后的224x224图像）", use_container_width=True)
            with cols[1]:
                if attn_overlay is not None:
                    st.subheader("稀疏注意力叠加图")
                    st.image(attn_overlay, caption="稀疏注意力图", use_container_width=True)
            if se_attn_overlay is not None:
                with cols[2]:
                    st.subheader("SE注意力叠加图")
                    st.image(se_attn_overlay, caption="SE注意力图叠加", use_container_width=True)
                st.subheader("SE注意力热力图")
                fig_se, ax_se = plt.subplots(figsize=(5, 5))
                im_se = ax_se.imshow(se_attn_heatmap.numpy(), cmap='jet')
                fig_se.colorbar(im_se)
                ax_se.axis('off')
                st.pyplot(fig_se)    
            if se_weights:
                try:
                    w = se_weights[-1].numpy()  # 使用最后一个 SE 块的权重
                    font = FontProperties(fname='SimHei.ttf')
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.bar(range(len(w)), w)
                    ax.set_title("SE通道权重", fontproperties=font)
                    ax.set_xlabel("通道", fontproperties=font)
                    ax.set_ylabel("权重", fontproperties=font)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"无法显示通道权重: {str(e)}")

        # AI Explanation Section
        if 'explainer' in st.session_state and st.session_state.get('explainer_loaded', False):
            st.markdown("---")
            st.subheader("🤖 大语言模型辅助诊断")
            with st.spinner("大语言模型诊断中..."):
                try:
                    diagnosis = (labels[pred_idx], probs[pred_idx].item())
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    image_base64 = base64.b64encode(buffered.getvalue()).decode()
                    explanation = st.session_state.explainer.generate_explanation(diagnosis, image_base64)
                    st.markdown("""
                        <style>
                            .report-box {
                                border: 1px solid #e0e0e0;
                                border-radius: 8px;
                                padding: 15px;
                                margin-top: 10px;
                                background-color: #f8f9fa;
                                font-family: sans-serif;
                                line-height: 1.6;
                            }
                        </style>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                        <div class="report-box">
                            {explanation.replace('\n', '<br>')}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # PDF Generation
                    buf_input = BytesIO()
                    img.save(buf_input, format="PNG")
                    buf_input.seek(0)
                    buf_conf = BytesIO()
                    fig_conf.savefig(buf_conf, format="PNG", bbox_inches='tight')
                    buf_conf.seek(0)
                    buf_cam = BytesIO()
                    if cam_overlay is not None:
                        cam_img = Image.fromarray((cam_overlay * 255).astype('uint8'))
                        cam_img.save(buf_cam, format="PNG")
                        buf_cam.seek(0)
                    def write_tmp(buf, suffix=".png"):
                        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                        tmp.write(buf.getvalue())
                        tmp.flush()
                        return tmp.name
                    input_path = write_tmp(buf_input)
                    conf_path = write_tmp(buf_conf)
                    cam_path = write_tmp(buf_cam) if cam_overlay is not None else None
                    pdf = FPDF(format='A4')
                    pdf.add_page()
                    pdf.set_margins(10, 10, 10)
                    pdf.set_auto_page_break(auto=True, margin=10)
                    pdf.add_font('SimHei', '', 'SimHei.ttf', uni=True)
                    pdf.set_font("SimHei", "", 14)
                    pdf.cell(0, 8, "病理诊断完整报告", ln=True, align="C")
                    pdf.ln(6)
                    img_w = 100
                    pdf.set_font("SimHei", "", 12)
                    pdf.cell(0, 6, "1. 上传原图", ln=True)
                    pdf.image(input_path, x=(pdf.w - img_w)/2, w=img_w)
                    pdf.ln(2)
                    pdf.set_font("SimHei", "", 10)
                    pdf.cell(0, 5, "图1: 上传的原始病理图像", align="C")
                    pdf.ln(4)
                    pdf.set_font("SimHei", "", 12)
                    pdf.cell(0, 6, "2. Grad-CAM 可视化", ln=True)
                    pdf.image(cam_path, x=(pdf.w - img_w)/2, w=img_w)
                    pdf.ln(2)
                    pdf.set_font("SimHei", "", 10)
                    pdf.cell(0, 5, "图2: Grad-CAM 可视化结果", align="C")
                    pdf.ln(4)
                    pdf.set_font("SimHei", "", 12)
                    pdf.cell(0, 6, "3. 置信度分布", ln=True)
                    if cam_path:
                        pdf.image(conf_path, x=(pdf.w - img_w)/2, w=img_w)
                        pdf.ln(2)
                        pdf.set_font("SimHei", "", 10)
                        pdf.cell(0, 5, "图3: 置信度分布图", align="C")
                        pdf.ln(4)
                    pdf.set_font("SimHei", "", 12)
                    pdf.cell(0, 6, "4. 诊断结果", ln=True)
                    pdf.set_font("SimHei", "", 10)
                    pdf.multi_cell(0, 5, f"类别：{pred_label}\n置信度：{pred_conf:.1f}%")
                    pdf.ln(4)
                    pdf.set_font("SimHei", "", 12)
                    pdf.cell(0, 6, "5. AI 生成的病理报告（摘要）", ln=True)
                    pdf.set_font("SimHei", "", 10)
                    for line in explanation.split("\n")[:8]:
                        pdf.multi_cell(0, 5, line)
                    pdf_bytes = pdf.output(dest='S')
                    pdf_buffer = BytesIO(pdf_bytes.encode('latin-1'))
                    pdf_buffer.seek(0)
                    st.download_button("⬇️ Download PDF", data=pdf_buffer, file_name=f"{pred_label}.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Failed to generate AI explanation: {str(e)}")
        else:
            st.info("AI explanation service is not available.")

    except Exception as e:
        st.error(f"An error occurred during image processing or diagnosis: {str(e)}")
        st.exception(e)

# --- Model Information Expander ---
with st.expander("模型系统消息"):
    st.markdown("""
    **模型架构:** NylonFuseNet
    * **深度:** [2, 2, 12, 2]
    * **维度:** [128, 256, 512, 1024]
    * **分类:** 8
    * **数据集:** Kvasir V2
    """)
    if st.button("验证模型状态"):
        if 'model' in st.session_state:
            st.write(f"设备上的模型: `{next(st.session_state.model.parameters()).device}`")
            st.write(f"参数: {sum(p.numel() for p in st.session_state.model.parameters())}")
        else:
            st.error("没有找到模型")

# --- Footer ---
st.divider()
st.caption("Kvasir Pathology Diagnosis System - Version 1.0")