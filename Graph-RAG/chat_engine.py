"""
SmartDoc AI — Chat Engine.

Manages multi-turn chat sessions with RAG-augmented streaming responses.
Uses GraphRAG local search for context retrieval.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv

from source.config import settings

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ChatSession:
    id: str
    domain: str
    title: str = "New Chat"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    messages: List[ChatMessage] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> ChatMessage:
        msg = ChatMessage(role=role, content=content)
        self.messages.append(msg)
        # Auto-set title from first user message
        if role == "user" and self.title == "New Chat":
            self.title = content[:50] + ("..." if len(content) > 50 else "")
        return msg


class SessionManager:
    """In-memory chat session manager."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ChatSession] = {}

    def create(self, domain: str) -> ChatSession:
        session = ChatSession(id=str(uuid.uuid4()), domain=domain)
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def list_all(self) -> List[ChatSession]:
        return sorted(
            self._sessions.values(),
            key=lambda s: s.created_at,
            reverse=True,
        )


# Global session manager
sessions = SessionManager()


async def chat_stream(session: ChatSession, user_message: str) -> AsyncGenerator[str, None]:
    """
    Stream a chat response token by token.

    1. Get context from GraphRAG local search
    2. Build prompt with conversation history + context
    3. Stream response from LLM via OpenRouter
    """
    # Record user message
    session.add_message("user", user_message)

    # 1. Get context from GraphRAG
    context = ""
    try:
        from source.query_engine import query_local, is_domain_ready

        if is_domain_ready(session.domain):
            result = await query_local(
                domain=session.domain,
                query=user_message,
                community_level=2,
                response_type="Single Paragraph",
            )
            if result.get("response") and not result.get("error"):
                context = result["response"]
    except Exception as e:
        logger.warning("Failed to get GraphRAG context: %s", e)

    # 2. Build prompt
    system_prompt = _build_system_prompt(context)
    messages = _build_messages(session, system_prompt)

    # 3. Stream response from LLM
    full_response = ""

    try:
        import httpx

        api_key = settings.llm_api_key
        base_url = settings.llm_base_url
        model = settings.llm_model

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 2048,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            full_response += token
                            yield token
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue

    except ImportError:
        # Fallback: non-streaming with requests
        full_response = await _fallback_generate(messages)
        yield full_response

    except Exception as e:
        logger.exception("Chat stream error")
        error_msg = f"Xin lỗi, đã xảy ra lỗi: {str(e)}"
        full_response = error_msg
        yield error_msg

    # Record assistant response
    if full_response:
        session.add_message("assistant", full_response)


def _build_system_prompt(context: str) -> str:
    """Build system prompt with optional RAG context."""
    base = (
        "Bạn là SmartDoc AI — trợ lý thông minh giúp người dùng truy vấn và tìm hiểu "
        "thông tin từ các tài liệu đã được đưa vào hệ thống. "
        "Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc."
    )

    if context:
        base += (
            "\n\nDưới đây là thông tin từ Knowledge Base liên quan đến câu hỏi:\n"
            f"---\n{context}\n---\n"
            "Hãy sử dụng thông tin trên để trả lời. Nếu không đủ, bổ sung bằng kiến thức chung."
        )
    else:
        base += (
            "\n\nHiện tại không có context từ Knowledge Base. "
            "Hãy trả lời dựa trên kiến thức chung và gợi ý người dùng upload tài liệu nếu cần."
        )

    return base


def _build_messages(session: ChatSession, system_prompt: str) -> List[Dict[str, str]]:
    """Build messages list for LLM API call."""
    messages = [{"role": "system", "content": system_prompt}]

    # Include recent conversation history (last 10 messages)
    recent = session.messages[-10:]
    for msg in recent:
        messages.append({"role": msg.role, "content": msg.content})

    return messages


async def _fallback_generate(messages: List[Dict[str, str]]) -> str:
    """Fallback: non-streaming generation using requests."""
    try:
        import requests as req

        api_key = settings.llm_api_key
        base_url = settings.llm_base_url
        model = settings.llm_model

        response = req.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            timeout=120,
        )

        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        else:
            return f"Lỗi API ({response.status_code}): {response.text[:200]}"

    except Exception as e:
        return f"Lỗi kết nối LLM: {str(e)}"
