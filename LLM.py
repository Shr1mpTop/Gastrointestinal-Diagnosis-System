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
            请严格按以下结构说明：
            1. 病理特征描述：（100字）
            2. 典型临床表现：（100字）
            3. 鉴别诊断建议：（100字）
            4. 后续检查建议：（100字）
            """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业病理科医生，需要用通俗易懂的中文解释诊断结果"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"AI解释生成失败: {str(e)}")
            return "无法获取AI解释"

    