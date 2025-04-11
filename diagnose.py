from datetime import datetime
import streamlit as st
import torch
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MedMamba import VSSM
from io import BytesIO
import base64
from LLM import DiagnosisExplainer
from torchvision import transforms

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型配置
@st.cache_resource
def load_model():
    model = VSSM(depths=[2, 2, 12, 2], dims=[128,256,512,1024], num_classes=8).to(device)
    model.load_state_dict(torch.load('medmambaNet.pth', map_location=device))
    model.eval()
    return model.to(device)

# 预处理管道
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载类别标签
with open('class_indices.json', 'r') as f:
    class_indict = json.load(f)
labels = [v for _, v in class_indict.items()]

# 界面布局
st.set_page_config(page_title="Kvasir病理诊断系统", layout="wide")
st.title("Kvasir内镜图像智能诊断系统")

# 初始化LLM解释器
from LLM import DiagnosisExplainer
if 'explainer' not in st.session_state:
    st.session_state.explainer = DiagnosisExplainer("6b7a963f-0952-4338-8e3e-29460040f0bf")

# 初始化会话状态
if 'model' not in st.session_state:
    try:
        st.session_state.model = load_model()
        st.success("模型加载成功！")
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")

# 侧边栏说明
with st.sidebar:
    st.header("使用说明")
    st.markdown("""
    1. 上传内镜图像（JPEG/PNG格式）
    2. 系统自动进行病理分析
    3. 查看诊断结果和置信度
    4. 点击"生成详细解释"获取AI分析
    """)
    st.divider()
    if st.button("🛠️ 系统健康检查"):
        with st.status("执行全面诊断...", expanded=True) as status:
            check_results = []
            
            # 验证核心依赖
            try:
                import torch, streamlit, PIL, numpy
                check_results.append(("核心依赖", "✅ 通过", f"Torch {torch.__version__}"))
            except ImportError as e:
                check_results.append(("核心依赖", "❌ 失败", f"缺失模块: {str(e)}"))

            # 验证模型加载
            try:
                model = load_model()
                check_results.append(("病理模型", "✅ 通过", f"参数数量: {len(list(model.parameters()))}"))
            except Exception as e:
                check_results.append(("病理模型", "❌ 失败", f"加载错误: {str(e)}"))

            # 验证API连接
            try:
                explainer = DiagnosisExplainer("6b7a963f-0952-4338-8e3e-29460040f0bf")
                check_results.append(("AI解释服务", "✅ 通过", "端点响应正常"))
            except Exception as e:
                check_results.append(("AI解释服务", "❌ 失败", f"连接错误: {str(e)}"))

            # 显示检查结果
            status.update(label="诊断完成", state="complete")
            st.table(pd.DataFrame(check_results, columns=["模块", "状态", "详情"]))

# 主界面
uploaded_file = st.file_uploader("上传内镜图像", type=["jpg", "jpeg", "png"])
col1, col2 = st.columns(2)

if uploaded_file is not None:
    try:
        # 图像预处理
        img = Image.open(uploaded_file).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # 模型推理
        with torch.no_grad():
            output = st.session_state.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0] * 100

        # 初始化解释器
        if 'explainer' not in st.session_state:
            try:
                st.session_state.explainer = DiagnosisExplainer("6b7a963f-0952-4338-8e3e-29460040f0bf")
            except Exception as e:
                st.error(f"系统初始化失败: {str(e)}")
                st.stop()  # 使用st.stop()替代非法return
        
        with col1:
            st.image(img, caption="输入图像", use_container_width=True)
            
        with col2:
            # 显示诊断结果
            pred_idx = torch.argmax(probs).item()
            st.subheader(f"诊断结果: {labels[pred_idx]}")
            st.metric(label="置信度", value=f"{probs[pred_idx]:.1f}%")
        
        # 增强版结果展示
        with st.expander("📊 详细诊断分析", expanded=True):
            tab1, tab2 = st.tabs(["置信度分布","临床指南"])
            y_pos = np.arange(len(labels))
            with tab1:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.barh(y_pos, probs.cpu().numpy(), color='#1f77b4')
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                ax.set_title("Probability")
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels)
                st.pyplot(fig)
                            
            with tab2:
                st.markdown("""
                **临床处理指南**
                - 立即处理指征: 置信度 >85%
                - 建议会诊范围: 50%-85%
                - 定期复查建议: <50%
                """) 
        
        # 自动生成AI解释
        with st.spinner("AI分析中..."):
            try:
                diagnosis = (labels[pred_idx], probs[pred_idx].item())
                # 将上传图片转换为base64
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                image_base64 = base64.b64encode(buffered.getvalue()).decode()

                explanation = st.session_state.explainer.generate_explanation(diagnosis, image_base64)
                # 在新行居中显示报告标题
                st.markdown("<h2 style='text-align: center;'>病理分析报告</h2>", unsafe_allow_html=True)

                # 使用 Markdown 解析展示解释内容
                st.markdown("""
                    <style>
                        .report-box {
                            border: 1px solid #e0e0e0;
                            border-radius: 10px;
                            padding: 20px;
                            margin: 20px 0;
                            background: #f8f9fa;
                        }
                    </style>
                """, unsafe_allow_html=True)

                with st.expander("📋 完整病理报告", expanded=True):
                    st.markdown(f"""
                        <div class="report-box">
                            {explanation.replace('\n', '<br>')}
                        </div>
                    """, unsafe_allow_html=True)
                    st.download_button("下载完整报告", explanation, file_name="diagnosis_report.md")
            except Exception as e:
                st.error(f"解释生成失败: {str(e)}")
        
    except Exception as e:
        st.error(f"处理错误: {str(e)}")

# 添加模型信息折叠栏
with st.expander("模型详细信息"):
    st.markdown("""
    **模型架构参数**
    - 基础维度: [128, 256, 512, 1024]
    - 深度配置: [2, 2, 12, 2]
    - 分类头: 8类别
    """)
    if st.button("验证模型权重"):
        try:
            state_dict = torch.load('medmambaNet.pth')
            st.write(f"权重文件包含 {len(state_dict)} 个参数")
        except:
            st.error("权重文件加载失败")

# 添加页脚
st.divider()
st.caption("医疗诊断系统 - 基于MedMamba架构 版本1.0")
