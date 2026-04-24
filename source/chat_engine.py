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
    role: str
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
        if role == "user" and self.title == "New Chat":
            self.title = content[:50] + ("..." if len(content) > 50 else "")
        return msg


class SessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, ChatSession] = {}
        self._storage_path = settings.index_path / "chat_sessions.json"
        self._load_from_storage()

    def _load_from_storage(self):
        if not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for sess_id, sess_data in data.items():
                    messages = [
                        ChatMessage(role=m["role"], content=m["content"], timestamp=m.get("timestamp", ""))
                        for m in sess_data.get("messages", [])
                    ]
                    session = ChatSession(
                        id=sess_id,
                        domain=sess_data["domain"],
                        title=sess_data.get("title", "New Chat"),
                        created_at=sess_data.get("created_at", ""),
                        messages=messages
                    )
                    self._sessions[sess_id] = session
            logger.info("Loaded %d chat sessions from disk", len(self._sessions))
        except Exception as e:
            logger.error("Failed to load chat sessions: %s", e)

    def _save_to_storage(self):
        try:
            data = {}
            for sess_id, s in self._sessions.items():
                data[sess_id] = {
                    "domain": s.domain,
                    "title": s.title,
                    "created_at": s.created_at,
                    "messages": [
                        {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                        for m in s.messages
                    ]
                }
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to save chat sessions: %s", e)

    def create(self, domain: str) -> ChatSession:
        session = ChatSession(id=str(uuid.uuid4()), domain=domain)
        self._sessions[session.id] = session
        self._save_to_storage()
        return session

    def get(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save_to_storage()
            return True
        return False

    def list_all(self) -> List[ChatSession]:
        return sorted(
            self._sessions.values(),
            key=lambda s: s.created_at,
            reverse=True,
        )

    def notify_update(self):
        self._save_to_storage()


sessions = SessionManager()


async def chat_stream(session: ChatSession, user_message: str) -> AsyncGenerator[str, None]:
    session.add_message("user", user_message)
    sessions.notify_update()

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

    system_prompt = _build_system_prompt(context)
    messages = _build_messages(session, system_prompt)

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
        full_response = await _fallback_generate(messages)
        yield full_response
    except Exception as e:
        logger.exception("Chat stream error")
        error_msg = f"Xin lỗi, đã xảy ra lỗi: {str(e)}"
        full_response = error_msg
        yield error_msg

    if full_response:
        session.add_message("assistant", full_response)
        sessions.notify_update()


def _build_system_prompt(context: str) -> str:
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
    messages = [{"role": "system", "content": system_prompt}]
    for msg in session.messages[-10:]:
        messages.append({"role": msg.role, "content": msg.content})
    return messages


async def _fallback_generate(messages: List[Dict[str, str]]) -> str:
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
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Lỗi API ({response.status_code}): {response.text[:200]}"
    except Exception as e:
        return f"Lỗi kết nối LLM: {str(e)}"
