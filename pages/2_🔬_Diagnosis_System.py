# pages/1_🔬_Diagnosis_System.py
import streamlit as st
import torch
from PIL import Image
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# MedMamba and DiagnosisExplainer are not directly needed here if loaded in app.py
# from MedMamba import VSSM # Not needed if model is in session state
# from LLM import DiagnosisExplainer # Not needed if explainer is in session state
from io import BytesIO
import base64
from torchvision import transforms

# --- Configuration ---
# Device configuration (can inherit from app.py or redefine if needed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Page Title ---
st.title("Kvasir Diagnosis & Analysis")

# --- Check if Model and Explainer are Loaded ---
if 'model' not in st.session_state or not st.session_state.get('model_loaded', False):
    st.error("Pathology model is not loaded. Please check the main application logs or system status on the homepage.")
    st.stop() # Stop execution of this page if model isn't ready

if 'explainer' not in st.session_state or not st.session_state.get('explainer_loaded', False):
    st.warning("AI Explainer is not loaded. Diagnosis will work, but explanations will be unavailable.")
    # Don't stop, allow diagnosis without explanation

# --- Preprocessing Pipeline ---
# (Keep this specific to the diagnosis page as it's used for processing uploads)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Load Class Labels ---
# (Keep this specific to the diagnosis page as it's needed for interpreting results)
try:
    with open('class_indices.json', 'r') as f:
        class_indict = json.load(f)
    labels = [v for _, v in class_indict.items()]
except FileNotFoundError:
    st.error("Error: 'class_indices.json' not found. Cannot map predictions to labels.")
    st.stop()
except Exception as e:
    st.error(f"Error loading class labels: {str(e)}")
    st.stop()

# --- Sidebar for Diagnosis Page ---
with st.sidebar:
    st.header("Usage Instructions")
    st.markdown("""
    1. Upload an endoscopic image (JPEG/PNG).
    2. The system will automatically analyze the pathology.
    3. Review the diagnosis result and confidence score.
    4. Expand the sections below for detailed analysis and AI-generated report.
    """)
    st.divider()
    # --- Health Check (Moved Here) ---
    if st.button("🛠️ System Health Check"):
        # Reuse the health check logic if you need it here
        # Make sure it checks session_state variables now
        with st.status("Running diagnostics...", expanded=True) as status:
            check_results = []
            # Check core dependencies (less critical now, but can keep)
            try:
                import torch, PIL, numpy
                check_results.append(("Core Libraries", "✅ OK", f"Torch {torch.__version__}"))
            except ImportError as e:
                check_results.append(("Core Libraries", "❌ Failed", f"Missing: {str(e)}"))

            # Check model in session state
            if 'model' in st.session_state and st.session_state.get('model_loaded', False):
                 check_results.append(("Pathology Model", "✅ Loaded", f"Device: {device}"))
            else:
                 check_results.append(("Pathology Model", "❌ Not Loaded", "Check homepage status"))

            # Check explainer in session state
            if 'explainer' in st.session_state and st.session_state.get('explainer_loaded', False):
                 check_results.append(("AI Explainer", "✅ Initialized", "Ready"))
            else:
                 check_results.append(("AI Explainer", "❌ Not Initialized", "Check homepage status / API Key"))

            # Display results
            status.update(label="Diagnostics Complete", state="complete")
            st.table(pd.DataFrame(check_results, columns=["Component", "Status", "Details"]))


# --- Main Interface ---
uploaded_file = st.file_uploader("Upload Endoscopic Image", type=["jpg", "jpeg", "png"])
col1, col2 = st.columns(2)

if uploaded_file is not None:
    try:
        # Image Preprocessing
        img = Image.open(uploaded_file).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)

        # Model Inference (using model from session state)
        with torch.no_grad():
            # Access model from session state
            output = st.session_state.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0] * 100

        # Display Input Image
        with col1:
            st.image(img, caption="Input Image", use_container_width=True)

        # Display Diagnosis Result
        with col2:
            pred_idx = torch.argmax(probs).item()
            st.subheader(f"Diagnosis Result: {labels[pred_idx]}")
            st.metric(label="Confidence", value=f"{probs[pred_idx]:.1f}%")

        # Enhanced Results Display
        with st.expander("📊 Detailed Diagnostic Analysis", expanded=True):
            tab1, tab2 = st.tabs(["Confidence Distribution", "Clinical Guidelines"])
            y_pos = np.arange(len(labels))
            with tab1:
                fig, ax = plt.subplots(figsize=(10, 5)) # Adjusted size slightly
                ax.barh(y_pos, probs.cpu().numpy(), color='#1f77b4', align='center')
                plt.rcParams['axes.unicode_minus'] = False
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels)
                ax.invert_yaxis()  # Labels read top-to-bottom
                ax.set_xlabel("Probability (%)")
                ax.set_title("Predicted Probability per Class")
                st.pyplot(fig)

            with tab2:
                st.markdown("""
                **Suggested Clinical Handling (Example Guidelines)**
                * **High Confidence (>85%):** Consider immediate action/consultation based on finding.
                * **Moderate Confidence (50%-85%):** Recommend further review or specialist consultation.
                * **Low Confidence (<50%):** Suggest monitoring or routine follow-up.
                * *Disclaimer: These are example guidelines and not medical advice.*
                """)

        # AI Explanation Section
        # Check if explainer is available before trying to use it
        if 'explainer' in st.session_state and st.session_state.get('explainer_loaded', False):
            st.markdown("---") # Separator
            st.subheader("🤖 AI-Generated Pathology Report")
            with st.spinner("AI is analyzing the findings..."):
                try:
                    diagnosis = (labels[pred_idx], probs[pred_idx].item())
                    # Convert image to base64 for the explainer
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    image_base64 = base64.b64encode(buffered.getvalue()).decode()

                    # Use explainer from session state
                    explanation = st.session_state.explainer.generate_explanation(diagnosis, image_base64)

                    # Display Report using Markdown with styling
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
                            .report-box h3 { margin-top: 0; }
                        </style>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                        <div class="report-box">
                            {explanation.replace('\n', '<br>')}
                        </div>
                    """, unsafe_allow_html=True)

                    st.download_button(
                        label="⬇️ Download Full Report (.md)",
                        data=explanation,
                        file_name=f"diagnosis_report_{labels[pred_idx]}.md",
                        mime="text/markdown"
                    )
                except Exception as e:
                    st.error(f"Failed to generate AI explanation: {str(e)}")
        else:
             st.info("AI explanation service is not available.")

    except Exception as e:
        st.error(f"An error occurred during image processing or diagnosis: {str(e)}")
        st.exception(e) # Provides more detailed traceback for debugging

# --- Model Information Expander ---
with st.expander("Model & System Information"):
    st.markdown("""
    **Model Architecture:** MedMamba (VSSM)
    * **Depths:** [2, 2, 12, 2]
    * **Dimensions:** [128, 256, 512, 1024]
    * **Number of Classes:** 8
    * **Dataset:** Kvasir V2 (likely basis)
    """)
    # Simplified weight check - direct loading might be slow/redundant here
    if st.button("Verify Model State"):
        if 'model' in st.session_state:
            st.write(f"Model loaded on device: `{next(st.session_state.model.parameters()).device}`")
            st.write(f"Number of parameters (approx): {sum(p.numel() for p in st.session_state.model.parameters())}")
        else:
            st.error("Model not found in session state.")

# --- Footer ---
st.divider()
st.caption("Kvasir Pathology Diagnosis System - Version 1.0")