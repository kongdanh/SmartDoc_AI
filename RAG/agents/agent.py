import os
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()


class Agent:

    def __init__(self, tool_registry):

        api_key = os.getenv("GEMINI_API_KEY")

        self.client = genai.Client(api_key=api_key)

        self.tools = tool_registry

    def think(self, question, scratchpad):

        tool_list = self.tools.list_tools()

        prompt = f"""
Bạn là một tác nhân AI tự động (autonomous AI agent).

Bạn có thể sử dụng các công cụ.

Các công cụ có sẵn:
{tool_list}

Hãy sử dụng định dạng sau:

Thought: suy luận
Action: tên công cụ
Action Input: dữ liệu đầu vào
Observation: kết quả

Câu hỏi: {question}

{scratchpad}

Thought:
"""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text