from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
import google.genai as genai

load_dotenv()


class Generator:
    def __init__(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in environment variables.")

        self.client = genai.Client(api_key=api_key)
        self.default_model = os.getenv("GEMINI_GENERATE_MODEL", "gemini-2.5-flash")
        self.fallback_answer = (
            "Hiện tại tôi chưa thể trả lời câu hỏi này, "
            "Bạn vui lòng liên hệ với Account hoặc TSM/MKT để được giải đáp thắc mắc. "
            "Xin cảm ơn"
        )

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""
Bạn là trợ lý AI chuyên nghiệp của ứng dụng Audit App, hỗ trợ đội ngũ kinh doanh (Account, TSM, MKT) tra cứu thông tin và tư vấn nghiệp vụ.

NGUYÊN TẮC TRẢ LỜI:
1. Phân tích kỹ câu hỏi, sau đó tổng hợp thông tin từ tất cả các đoạn tài liệu liên quan.
2. Ưu tiên thông tin trong tài liệu; nếu tài liệu chưa đủ, bổ sung bằng kiến thức chuyên môn phù hợp với ngữ cảnh.
3. Trả lời bằng tiếng Việt, rõ ràng, chuyên nghiệp và đủ chi tiết để người đọc hiểu hoàn toàn.
4. Nếu câu hỏi có nhiều khía cạnh, trình bày có cấu trúc (danh sách, bước thực hiện, v.v.).
5. Giữ nguyên citation dạng [1], [2]... khi trích dẫn từ tài liệu.
6. CHỈ trả lời "{self.fallback_answer}" khi câu hỏi hoàn toàn không liên quan đến lĩnh vực kinh doanh/audit và không có bất kỳ thông tin nào để hỗ trợ.

[CÂU HỎI]
{query}

[NỘI DUNG TÀI LIỆU]
{context}

Yêu cầu output JSON (không bọc trong markdown):
{{
  "answer": "câu trả lời đầy đủ, chi tiết tại đây"
}}
""".strip()

    def generate(self, query: str, context: str, model_name: str | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        model = model_name or self.default_model
        prompt = self._build_prompt(query, context)

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
        )

        text = getattr(response, "text", "") or ""
        text = text.strip()

        # Strip markdown code fences if present (e.g. ```json ... ```)
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*\n?', '', text)
            text = re.sub(r'\n?```\s*$', '', text)
            text = text.strip()

        parsed: Dict[str, Any]
        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                parsed = {"answer": text}
        except Exception:
            parsed = {"answer": self.fallback_answer}

        answer_text = str(parsed.get("answer", "") or "").strip()
        if not answer_text:
            parsed["answer"] = self.fallback_answer

        usage = getattr(response, "usage_metadata", None)

        token_info = {
            "model": model,
            "prompt_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
            "response_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_token_count", 0) if usage else 0,
        }

        return parsed, token_info