import streamlit as st
from openai import OpenAI
from typing import Tuple
import os

class DiagnosisExplainer:
    def __init__(self, api_key: str, base_url="https://ark.cn-beijing.volces.com/api/v3", model="ep-20250411130043-bpkcr"):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model

    def generate_explanation(self, diagnosis: Tuple[str, float], image_base64: str = None) -> str:
        """生成病理诊断的详细解释
        Args:
            diagnosis: (病理类型, 置信度) 元组
            image_base64: 可选的基础64编码图像
        """
        prompt = f"""
            诊断结果：{diagnosis[0]}
            置信度：{diagnosis[1]:.1f}%
            请分别用 80 至 120 字，清晰扼要地阐述以下四个部分：
            1. **病理特征描述**：该疾病在组织学或病理学上的主要表现。  
            2. **典型临床表现**：患者可能出现的主要症状与体征。  
            3. **鉴别诊断建议**：根据置信度高低，给出两种情景下的鉴别诊断或进一步判断思路。  
            4. **后续检查建议**：针对当前结果，推荐的下一步检查或复查方案。
            """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的病理科医生，请用通俗易懂的中文，面向非医学专业的患者或家属，详细解读诊断结果。请严格按照用户给出的结构输出，标题要居中加粗（Markdown 风格），正文简洁连贯，不要附加其他无关内容。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"AI解释生成失败: {str(e)}")
            return "无法获取AI解释"

    