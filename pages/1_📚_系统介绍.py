# pages/1_📚_GI_Conditions_Overview.py

import streamlit as st
import json
import pandas as pd
import plotly.express as px
from PIL import Image # To check for image files potentially

# --- Configuration ---
st.set_page_config(page_title="训练数据集概览", layout="wide") # Set config for this page
st.title("📚 胃肠道疾病参考指南")
st.markdown("---")
st.markdown("""
本页面提供Kvasir诊断系统基于v2数据集可识别的胃肠道疾病及解剖标志的背景知识，
理解这些信息有助于更好理解诊断结果。
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
        "染色息肉", "染色边缘", "食管炎",
        "正常盲肠", "正常幽门", "正常 Z 线",
        "息肉", "溃疡性结肠炎"
    ]
    st.warning("Using fallback condition names.")
except Exception as e:
    st.error(f"Error loading class labels: {str(e)}")
    condition_names = []
    st.stop()


image_paths = {
    "染色息肉": "images/dyed-lifted-polyps.jpg",
    "染色边缘": "images/dyed-resection-margins.jpg",
    "食管炎": "images/esophagitis.jpg",
    "正常盲肠": "images/normal-cecum.jpg",
    "正常幽门": "images/normal-pylorus.jpg",
    "正常 Z 线": "images/normal-z-line.jpg",
    "息肉": "images/polyps.jpg",
    "溃疡性结肠炎": "images/ulcerative-colitis.jpg"
}

# --- 病理描述 ---
descriptions = {
    "染色息肉": "息肉是异常增生组织。'染色抬举'指息肉切除术中使用靛胭脂等染料注射至息肉下方，使其从肠壁抬起的技术，便于更安全完整地切除。",
    "染色边缘": "切除息肉或病变后，对切除边缘进行染色标记，帮助内镜医师评估是否完整切除。清晰的切缘对预防复发至关重要。",
    "食管炎": "食管炎症，多由胃食管反流病(GERD)引起，内镜下可见红斑、糜烂或溃疡等表现。",
    "正常盲肠": "盲肠作为结肠起始部，连接回肠与升结肠。正常盲肠可见阑尾开口及回盲瓣等解剖结构。",
    "正常幽门": "幽门是胃出口的肌性阀门，控制食糜进入十二指肠。正常幽门显示完整无溃疡的括约肌结构。",
    "正常 Z 线": "Z线是食管鳞状上皮与胃柱状上皮的交界，正常呈清晰规整的锯齿状，无肠化生表现。",
    "息肉": "胃肠道常见隆起性病变，结肠息肉尤其多见。腺瘤性息肉有癌变风险，需内镜下切除。",
    "溃疡性结肠炎": "慢性炎症性肠病，特征为结肠连续性炎症和溃疡形成，常见血便、腹痛等症状。"
}

# --- 临床背景 ---
context = {
    "染色息肉": "常见于结直肠癌筛查中的息肉切除手术",
    "染色边缘": "息肉切除术的质量控制关键指标",
    "食管炎": "全球常见疾病，多与胃食管反流病相关",
    "正常盲肠": "结肠镜检查完整性的解剖标志",
    "正常幽门": "上消化道内镜检查的重要解剖标志",
    "正常 Z 线": "评估胃食管反流病及巴雷特食管的关键标志",
    "息肉": "随年龄增长发病率升高，结肠镜筛查主要目标",
    "溃疡性结肠炎": "炎症性肠病主要类型，全球发病率持续上升"
}


# --- Display Conditions ---
st.header("Kvasir数据集中不同类别病症介绍")

for name in condition_names:
    with st.expander(f"**{name.replace('-', ' ').title()}**", expanded=False):
        col1, col2 = st.columns([2, 1]) # Text column wider than image column

        with col1:
            st.markdown(f"**病理描述:** {descriptions.get(name, '暂无描述信息')}")
            st.markdown(f"**临床意义:** {context.get(name, '暂无上下文信息')}")
            st.markdown(f"*[免责声明：描述内容经过简化处理，仅用于教学目的]*")

        with col2:
            img_path = image_paths.get(name)
            if img_path:
                try:
                    # Attempt to open the image to check if it exists
                    image = Image.open(img_path)
                    caption = f"{name.replace('-','')} 的详细描述图像"
                    st.image(image, caption=caption, use_container_width=True)
                except FileNotFoundError:
                    st.warning(f"Image not found at: {img_path}")
                    st.markdown(f"_(Image placeholder for {name})_")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            else:
                st.markdown(f"_(No image specified for {name})_")


# --- Global Perspective & Data Visualization ---
st.markdown("---")
st.header("全球视角与数据洞察")

# Textual context about global relevance
st.markdown("""
胃肠道疾病是全球重大健康负担：
* **结直肠癌(CRC):** 息肉等病变是CRC的前兆，CRC是全球最常见癌症之一。内镜筛查项目通过早期发现和切除息肉显著降低死亡率。截至2025年，多国仍在扩大筛查范围。
* **炎症性肠病(IBD):** 溃疡性结肠炎等疾病影响数百万人，在亚洲和南美新兴工业化国家发病率显著上升。针对肠道微生物组和靶向治疗的研究十分活跃。
* **食管疾病:** 胃食管反流病相关的食管炎全球常见，严重影响生活质量。早期发现巴雷特食管等癌前病变对预防腺癌至关重要。

**内镜AI技术:** 人工智能日益融入临床实践，帮助医生检测息肉和早期癌变，提升诊断效率和准确性。本系统即是该技术应用于Kvasir数据集的教学研究范例。
""")

# --- Simple Data Visualization (Kvasir Dataset Distribution - Example) ---
st.subheader("Kvasir v2 数据集不同类别数目分布")

# IMPORTANT: Replace this dummy data with the actual counts from your dataset analysis!
# These numbers are purely for demonstration purposes.
dummy_data = {
    "染色息肉": 1000,
    "染色边缘": 1000,
    "食管炎": 1000,
    "正常盲肠": 1000,
    "正常幽门": 1000,
    "正常 Z 线": 1000,
    "息肉": 1000,
    "溃疡性结肠炎": 1000
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
                     title="Kvasir v2 数据集类别分布示意图",
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
st.caption(f"Kvasir 病理诊断系统 - 疾病知识库 | 版本：2.0 | 最后更新：2025年4月")
