import streamlit as st
import torch
from grad_cam import GradCAM
from MedMamba import SS_Conv_SSM
from MedMamba import VSSM  # Assuming MedMamba.py is in the same directory or installed
from LLM import DiagnosisExplainer  # Assuming LLM.py is in the same directory or installed
import pandas as pd  # Keep pandas for the health check table if needed on homepage

# --- Configuration ---
# Set wide layout and page title for the entire app
st.set_page_config(
    page_title="Kvasir 病理诊断系统",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🔬"  # Optional: Add an icon
)

# --- Device Configuration (can be defined here or in diagnosis page) ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Model Loading (Cached Resource) ---
@st.cache_resource
def load_model():
    """Loads the VSSM model."""
    try:
        model = VSSM(depths=[2, 2, 12, 2], dims=[128, 256, 512, 1024], num_classes=8).to(device)
        # Ensure the path to your model weights is correct
        model_path = 'medmambaNet.pth'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model.to(device)
    except FileNotFoundError:
        st.error(f"Error: Model weight file '{model_path}' not found. Please ensure it's in the correct directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {str(e)}")
        return None

# --- LLM Explainer Initialization (Cached Resource) ---
@st.cache_resource
def initialize_explainer():
    """Initializes the Diagnosis Explainer."""
    try:
        # Replace with your actual API key or initialization logic
        explainer = DiagnosisExplainer("6b7a963f-0952-4338-8e3e-29460040f0bf")  # Make sure LLM.py is accessible
        return explainer
    except NameError:
        st.error("Error: DiagnosisExplainer class not found. Make sure LLM.py is available.")
        return None
    except Exception as e:
        st.error(f"Failed to initialize the AI Explainer: {str(e)}")
        return None

# --- Load Model and Explainer into Session State ---
# This ensures they are loaded once and available across pages
if 'model' not in st.session_state:
    model_instance = load_model()
    if model_instance:
        st.session_state.model = model_instance
        st.session_state.model_loaded = True
        # st.success("Pathology model loaded successfully!") # Optional feedback
    else:
        st.session_state.model_loaded = False
        st.error("Pathology model failed to load. Diagnosis functionality will be unavailable.")


if 'explainer' not in st.session_state:
    explainer_instance = initialize_explainer()
    if explainer_instance:
        st.session_state.explainer = explainer_instance
        st.session_state.explainer_loaded = True
        # st.success("AI Explainer initialized successfully!") # Optional feedback
    else:
        st.session_state.explainer_loaded = False
        st.error("AI Explainer failed to initialize. Explanation functionality will be unavailable.")

# --- Homepage Content ---
st.title("欢迎使用 Kvasir 内窥镜图像诊断系统")
st.markdown("---")

st.header("关于本系统")
st.markdown("""
本系统采用先进的**MedMamba**深度学习架构分析Kvasir数据集中的内窥镜图像，提供：
* 病理发现的自动分类
* 各项诊断的置信度评分
* 基于图像和诊断的AI生成病理解释

**请使用左侧边栏**进入主诊断系统。
""")

# Optional: Add an image to the homepage
# try:
#     st.image("path/to/your/homepage_image.jpg", caption="Intelligent Medical Image Analysis")
# except Exception:
#     st.warning("Homepage image not found.")


st.header("重要声明")
st.warning("""
**仅供研究及教学使用**
本系统是基于Kvasir数据集和MedMamba模型的实验性工具，
**不能替代**专业医疗建议、诊断或治疗。
遇到任何医疗问题请咨询合格医疗人员，诊断决策**切勿**仅依赖本系统输出。
""")

st.markdown("---")
st.caption("Kvasir Pathology Diagnosis System - Version 1.0")

# --- Sidebar Content for Homepage (Optional) ---
with st.sidebar:
    st.header("导航")
    st.markdown("请从上方选择页面")
    st.divider()
    st.header("系统状态")
    if st.session_state.get('model_loaded', False):
        st.success("✅ 病理模型已加载")
    else:
        st.error("❌ 病理模型加载失败")

    if st.session_state.get('explainer_loaded', False):
        st.success("✅ AI解释器已初始化")
    else:
        st.error("❌ AI解释器初始化失败")

    # You could add a simplified health check here if desired
    # st.button("Quick Check") # Example
