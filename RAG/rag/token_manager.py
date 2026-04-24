from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json


class TokenManager:
    def __init__(self) -> None:
        self.session_stats = {
            "total_prompt_tokens": 0,
            "total_output_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "queries": [],
        }

    def select_model(self, query: str):
        query = (query or "").strip()
        if len(query) > 800:
            return "gemini-2.5-pro", "Query dài, chọn model mạnh hơn"
        return "gemini-2.5-flash", "Query thông thường, chọn model nhanh"

    def estimate_cost(self, model: str, prompt_tokens: int, output_tokens: int) -> float:
        return 0.0

    def log_query(
        self,
        query: str,
        model_used: str,
        prompt_tokens: int,
        output_tokens: int,
        answer_preview: str = "",
    ):
        total_tokens = prompt_tokens + output_tokens
        cost = self.estimate_cost(model_used, prompt_tokens, output_tokens)

        log = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "model": model_used,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost,
            "answer_preview": (answer_preview or "")[:500],
        }

        self.session_stats["total_prompt_tokens"] += prompt_tokens
        self.session_stats["total_output_tokens"] += output_tokens
        self.session_stats["total_tokens"] += total_tokens
        self.session_stats["total_cost"] += cost
        self.session_stats["queries"].append(log)

        return log

    def print_session_summary(self) -> None:
        print("\n=== Session Summary ===")
        print(f"Prompt tokens: {self.session_stats['total_prompt_tokens']:,}")
        print(f"Output tokens: {self.session_stats['total_output_tokens']:,}")
        print(f"Total tokens:  {self.session_stats['total_tokens']:,}")
        print(f"Total cost:    ${self.session_stats['total_cost']:.6f}")

    def save_session_log(self, output_path: str = "logs/token_session.json") -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.session_stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )