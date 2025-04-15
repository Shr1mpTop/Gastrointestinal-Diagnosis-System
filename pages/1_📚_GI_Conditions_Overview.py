# pages/1_📚_GI_Conditions_Overview.py

import streamlit as st
import json
import pandas as pd
import plotly.express as px
from PIL import Image # To check for image files potentially

# --- Configuration ---
st.set_page_config(page_title="GI Conditions Overview", layout="wide") # Set config for this page
st.title("📚 Overview of Gastrointestinal Conditions")
st.markdown("---")
st.markdown("""
This page provides background information on the gastrointestinal (GI) conditions and landmarks
that the Kvasir Diagnosis System is trained to identify, based on the Kvasir v2 dataset.
Understanding these can help contextualize the diagnostic results.
""")

# --- Load Class Labels ---
# Make sure class_indices.json is accessible from the root or provide the correct path
labels_dict = {}
try:
    # Adjust path if necessary, e.g., ../class_indices.json if pages is a subdir and json is in root
    with open('class_indices.json', 'r') as f:
        class_indict = json.load(f)
    # Create a dictionary mapping name to index for easier lookup if needed
    labels_dict = {v: k for k, v in class_indict.items()}
    condition_names = list(labels_dict.keys()) # Get the names
except FileNotFoundError:
    st.error("Error: 'class_indices.json' not found. Cannot display condition details.")
    # Fallback names if file not found (adjust these to match your actual classes)
    condition_names = [
        "dyed-lifted-polyps", "dyed-resection-margins", "esophagitis",
        "normal-cecum", "normal-pylorus", "normal-z-line",
        "polyps", "ulcerative-colitis"
    ]
    st.warning("Using fallback condition names.")
except Exception as e:
    st.error(f"Error loading class labels: {str(e)}")
    condition_names = []
    st.stop()


image_paths = {
    "dyed-lifted-polyps": "images/dyed-lifted-polyps.jpg",
    "dyed-resection-margins": "images/dyed-resection-margins.jpg",
    "esophagitis": "images/esophagitis.jpg",
    "normal-cecum": "images/normal-cecum.jpg",
    "normal-pylorus": "images/normal-pylorus.jpg",
    "normal-z-line": "images/normal-z-line.jpg",
    "polyps": "images/polyps.jpg",
    "ulcerative-colitis": "images/ulcerative-colitis.jpg"
}

# --- Descriptions (Customize these as needed) ---
descriptions = {
    "dyed-lifted-polyps": "Polyps are abnormal growths. 'Dyed-lifted' refers to a technique used during removal (polypectomy) where dye (like indigo carmine) is injected under the polyp to lift it from the colon wall, making removal easier and safer. This image class captures the appearance after this procedure.",
    "dyed-resection-margins": "After removing a polyp or lesion, the edges ('margins') of the removal site are often dyed. This helps the endoscopist assess if the removal was complete (no abnormal tissue left behind). Clear margins are crucial to prevent recurrence.",
    "esophagitis": "Inflammation of the esophagus, the tube connecting the throat to the stomach. It can be caused by acid reflux (most common), infections, allergies, or certain medications. Endoscopically, it may appear as redness, swelling, erosions, or ulcers.",
    "normal-cecum": "The cecum is the beginning of the large intestine (colon), connecting the small intestine (ileum) to the colon. A 'normal cecum' view shows healthy tissue, often including the appendiceal orifice (opening to the appendix) and the ileocecal valve.",
    "normal-pylorus": "The pylorus is the muscular valve ('pyloric sphincter') at the exit of the stomach, controlling the flow of food into the small intestine (duodenum). A 'normal pylorus' view shows a healthy, functioning sphincter without inflammation, ulcers, or obstruction.",
    "normal-z-line": "The Z-line (or squamocolumnar junction) is the boundary in the lower esophagus where the squamous lining of the esophagus meets the columnar lining of the stomach. A 'normal Z-line' is sharp and regular, indicating no significant metaplasia (like Barrett's esophagus).",
    "polyps": "Abnormal growths protruding from the lining of the GI tract, most commonly the colon. While many are benign, some types (adenomatous polyps) can become cancerous over time. Endoscopic removal is key for colorectal cancer prevention.",
    "ulcerative-colitis": "A chronic inflammatory bowel disease (IBD) that causes inflammation and ulcers primarily in the lining of the large intestine (colon) and rectum. Symptoms include diarrhea, abdominal pain, and rectal bleeding. Endoscopy shows characteristic inflammation patterns."
}

# --- General Prevalence/Context (Customize with more specific data if available) ---
context = {
    "dyed-lifted-polyps": "Relates to polyp removal procedures, common in colorectal cancer screening.",
    "dyed-resection-margins": "Important quality marker in polypectomy procedures.",
    "esophagitis": "Very common, often linked to GERD (Gastroesophageal Reflux Disease), affecting millions globally.",
    "normal-cecum": "An anatomical landmark confirming complete colonoscopy.",
    "normal-pylorus": "Anatomical landmark in upper endoscopy.",
    "normal-z-line": "Important landmark for assessing conditions like GERD and Barrett's esophagus.",
    "polyps": "Colonic polyps are common, especially in older adults. Prevalence increases with age. Screening colonoscopy aims to detect and remove them.",
    "ulcerative-colitis": "A major type of IBD, with varying prevalence globally but increasing incidence in many regions. Affects millions worldwide."
}


# --- Display Conditions ---
st.header("Conditions Covered by the Kvasir Dataset")

for name in condition_names:
    with st.expander(f"**{name.replace('-', ' ').title()}**", expanded=False):
        col1, col2 = st.columns([2, 1]) # Text column wider than image column

        with col1:
            st.markdown(f"**Description:** {descriptions.get(name, 'No description available.')}")
            st.markdown(f"**General Context:** {context.get(name, 'No context available.')}")
            st.markdown(f"*[Disclaimer: Descriptions are simplified for educational purposes.]*")

        with col2:
            img_path = image_paths.get(name)
            if img_path:
                try:
                    # Attempt to open the image to check if it exists
                    image = Image.open(img_path)
                    st.image(image, caption=f"Illustrative example of {name.replace('-', ' ')}", use_container_width=True)
                except FileNotFoundError:
                    st.warning(f"Image not found at: {img_path}")
                    st.markdown(f"_(Image placeholder for {name})_")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            else:
                st.markdown(f"_(No image specified for {name})_")


# --- Global Perspective & Data Visualization ---
st.markdown("---")
st.header("Global Perspective & Data Insights")

# Textual context about global relevance
st.markdown("""
Gastrointestinal diseases represent a significant global health burden.
* **Colorectal Cancer (CRC):** Conditions like polyps are precursors to CRC, one of the most common cancers worldwide. Screening programs involving endoscopy aim to detect and remove polyps early, significantly reducing CRC mortality. As of 2025, many countries continue expanding their screening initiatives.
* **Inflammatory Bowel Disease (IBD):** Diseases like Ulcerative Colitis affect millions, with a noticeable increasing prevalence in newly industrialized countries across Asia and South America, adding to the high rates in North America and Europe. Research into microbiome and targeted therapies is very active.
* **Esophageal Conditions:** Esophagitis linked to GERD remains extremely common globally, impacting quality of life. Early detection and monitoring of related changes like Barrett's esophagus are crucial for preventing esophageal adenocarcinoma, the rates of which have risen in some populations over recent decades.

**AI in Endoscopy:** Artificial intelligence is increasingly integrated into clinical practice to aid endoscopists in detecting subtle lesions like polyps and early cancer, improving diagnostic accuracy and efficiency. This system is an example of such technology applied to the Kvasir dataset for educational and research exploration.
""")

# --- Simple Data Visualization (Kvasir Dataset Distribution - Example) ---
st.subheader("Kvasir v2 Dataset Class Distribution (Illustrative)")

# IMPORTANT: Replace this dummy data with the actual counts from your dataset analysis!
# These numbers are purely for demonstration purposes.
dummy_data = {
    "dyed-lifted-polyps": 1000,
    "dyed-resection-margins": 1000,
    "esophagitis": 1000,
    "normal-cecum": 1000,
    "normal-pylorus": 1000,
    "normal-z-line": 1000,
    "polyps": 1000,
    "ulcerative-colitis": 1000
}

# Use the condition_names list loaded at the beginning of the file
# Ensure condition_names is populated correctly before this point
if 'condition_names' in locals() and condition_names:
    # Create a dictionary with actual names and their counts from dummy_data
    data_counts = {name: dummy_data.get(name, 0) for name in condition_names}

    # Create a Pandas DataFrame
    df_dist = pd.DataFrame(list(data_counts.items()), columns=['Condition', 'Image Count'])
    df_dist = df_dist.sort_values(by='Image Count', ascending=False)

    if not df_dist.empty:
        # Create a bar chart using Plotly Express
        fig = px.bar(df_dist,
                     x='Condition',
                     y='Image Count',
                     title="Number of Images Per Class in Kvasir v2 (Example Data)",
                     labels={'Image Count': 'Number of Images', 'Condition': 'Condition/Finding'},
                     color='Condition', # Color bars by condition name
                     color_discrete_sequence=px.colors.qualitative.Pastel) # Use a pleasant color scheme
        fig.update_layout(xaxis_tickangle=-45, # Rotate labels if long
                          xaxis_title=None, # Hide x-axis title if redundant
                          yaxis_title="Number of Images",
                          showlegend=False) # Hide legend if coloring by category name is clear enough
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Note: This chart uses example data. Actual counts depend on the specific dataset version and splits used for training the model.")
    else:
        st.warning("Could not display dataset distribution chart. Condition data might be missing.")
else:
    st.warning("Condition names not loaded, cannot display distribution chart.")

# --- Final Disclaimer ---
st.markdown("---") # Adds a horizontal line separator

st.warning("""
**Important Disclaimer:** The information provided on this page and throughout this application
is intended for general informational and educational purposes only. It is **NOT** a substitute
for professional medical advice, diagnosis, or treatment.

Always seek the advice of your physician or other qualified health provider with any questions
you may have regarding a medical condition or interpreting endoscopic findings. Never disregard
professional medical advice or delay in seeking it because of something you have read or seen
in this application. Reliance on any information provided by this system is solely at your own risk.
""")

# --- Footer ---
st.divider() # Adds another visual separator line
st.caption(f"Kvasir Pathology Diagnosis System - Information Page | Content current as of April 2025") # Added a timestamp based on current context