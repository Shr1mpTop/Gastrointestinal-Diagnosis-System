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
    page_title="Kvasir Pathology Diagnosis System",
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
st.title("Welcome to the Kvasir Endoscopic Image Diagnosis System")
st.markdown("---")

st.header("About This System")
st.markdown("""
This system utilizes the advanced **MedMamba** deep learning architecture to analyze endoscopic images from the Kvasir dataset.
It provides:
* Automated classification of pathological findings.
* Confidence scores for each potential diagnosis.
* AI-generated explanations of the findings based on the image and diagnosis.

**Navigate using the sidebar on the left** to access the main Diagnosis System.
""")

# Optional: Add an image to the homepage
# try:
#     st.image("path/to/your/homepage_image.jpg", caption="Intelligent Medical Image Analysis")
# except Exception:
#     st.warning("Homepage image not found.")


st.header("Important Disclaimer")
st.warning("""
**For Research & Educational Purposes Only.**
This system is an experimental tool based on the Kvasir dataset and the MedMamba model.
It is **NOT** a substitute for professional medical advice, diagnosis, or treatment.
Always consult with a qualified healthcare provider for any medical concerns. Diagnostic decisions should **NEVER** be based solely on the output of this system.
""")

st.markdown("---")
st.caption("Kvasir Pathology Diagnosis System - Version 1.0")

# --- Sidebar Content for Homepage (Optional) ---
with st.sidebar:
    st.header("Navigation")
    st.markdown("Select a page above to get started.")
    st.divider()
    st.header("System Status")
    if st.session_state.get('model_loaded', False):
        st.success("✅ Pathology Model Loaded")
    else:
        st.error("❌ Pathology Model Failed")

    if st.session_state.get('explainer_loaded', False):
        st.success("✅ AI Explainer Initialized")
    else:
        st.error("❌ AI Explainer Failed")

    # You could add a simplified health check here if desired
    # st.button("Quick Check") # Example