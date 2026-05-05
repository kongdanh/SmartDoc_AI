"""
SmartDoc AI — Chat Engine.

PERFORMANCE FIX:
  - Replaced slow GraphRAG CLI query (~60-90s) with fast Standard RAG (~1-2s)
  - Added 15s timeout on context retrieval so chat never hangs
  - GraphRAG is still used via /api/compare-rag for side-by-side comparison
"""

from __future__ import annotations

import asyncio
import json
import logging
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
                    ChatMessage(
                        role=m["role"],
                        content=m["content"],
                        timestamp=m.get("timestamp", ""),
                    )
                    for m in sess_data.get("messages", [])
                ]
                self._sessions[sess_id] = ChatSession(
                    id=sess_id,
                    domain=sess_data["domain"],
                    title=sess_data.get("title", "New Chat"),
                    created_at=sess_data.get("created_at", ""),
                    messages=messages,
                )
            logger.info("Loaded %d chat sessions from disk", len(self._sessions))
        except Exception as e:
            logger.error("Failed to load chat sessions: %s", e)

    def _save_to_storage(self):
        try:
            data = {
                sess_id: {
                    "domain": s.domain,
                    "title": s.title,
                    "created_at": s.created_at,
                    "messages": [
                        {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                        for m in s.messages
                    ],
                }
                for sess_id, s in self._sessions.items()
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


# ─── Context retrieval (Standard RAG — fast) ─────────────────────
async def _get_context_fast(domain: str, query: str) -> str:
    """Get RAG context using Standard RAG (ChromaDB + local embeddings)."""
    try:
        from standard_rag import retrieve_context_only

        context = await asyncio.wait_for(
            asyncio.to_thread(retrieve_context_only, query, domain, 5),
            timeout=25.0,
        )
        return context or ""

    except asyncio.TimeoutError:
        logger.warning(
            "Standard RAG context retrieval timed out for domain '%s' — proceeding without context",
            domain,
        )
        return ""
    except Exception as e:
        logger.warning("Failed to get Standard RAG context: %s", e)
        return ""


async def chat_stream(session: ChatSession, user_message: str) -> AsyncGenerator[str, None]:
    session.add_message("user", user_message)
    sessions.notify_update()

    context = await _get_context_fast(session.domain, user_message)
    memory = _get_conversation_memory(session, max_turns=3)

    system_prompt = _build_system_prompt(context, memory)
    messages = _build_messages(session, system_prompt)

    full_response = ""
    try:
        import httpx

        headers = {
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": settings.llm_model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 4096,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{settings.llm_base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                
                # [FIX CHÍNH] Bắt lỗi từ LLM (Ví dụ: 429 Rate Limit) để in ra màn hình chat
                if response.status_code != 200:
                    err_body = await response.aread()
                    error_msg = f"Lỗi API từ LLM ({response.status_code}): {err_body.decode('utf-8')[:200]}"
                    full_response = error_msg
                    yield error_msg
                else:
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            token = (
                                data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
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
        error_msg = f"Xin lỗi, đã xảy ra lỗi kết nối: {str(e)}"
        full_response = error_msg
        yield error_msg

    if full_response:
        session.add_message("assistant", full_response)
        sessions.notify_update()
# ─── Helpers ─────────────────────────────────────────────────────

def _get_conversation_memory(session: ChatSession, max_turns: int = 3) -> str:
    """Extract last N conversation turns as memory context.
    
    Only returns Q&A pairs that exist. If there are 2 messages (1 pair),
    return just that. Works with any number from 1-3 pairs.
    
    Args:
        session: Chat session with message history
        max_turns: Max number of Q&A pairs to extract (default 3)
    
    Returns:
        Formatted conversation memory string, or empty string if no history
    """
    if len(session.messages) < 2:
        return ""
    
    total_pairs_available = len(session.messages) // 2
    pairs_to_take = min(max_turns, total_pairs_available)
    
    if pairs_to_take == 0:
        return ""
    
    messages_to_use = session.messages[-(pairs_to_take * 2):]
    
    memory_text = "Lịch sử cuộc trò chuyện gần nhất:\n"
    for msg in messages_to_use:
        if msg.role == "user":
            memory_text += f"Người dùng: {msg.content}\n"
        else:
            memory_text += f"Trợ lý: {msg.content}\n"
    
    return memory_text


def _build_system_prompt(context: str, memory: str = "") -> str:
    base = (
        "Bạn là SmartDoc AI — trợ lý thông minh giúp người dùng truy vấn và tìm hiểu "
        "thông tin từ các tài liệu đã được đưa vào hệ thống. "
        "Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc."
    )
    
    if memory:
        base += f"\n\n{memory}"
    
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
    """Synchronous fallback if httpx is unavailable."""
    try:
        import requests as req

        response = req.post(
            f"{settings.llm_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            json={
                "model": settings.llm_model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            timeout=120,
        )
        
        if response.status_code == 200:
            if not response.text:
                logger.error("Response body is empty")
                return "Lỗi: Phản hồi trống"
            try:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.error(f"Failed to parse response: {e}, body: {response.text[:500]}")
                return f"Lỗi parse response: {str(e)}"
        
        error_text = response.text[:200] if response.text else f"HTTP {response.status_code}"
        logger.error(f"API error ({response.status_code}): {error_text}")
        return f"Lỗi API ({response.status_code}): {error_text}"
    except Exception as e:
        logger.exception("Fallback generate error")
        return f"Lỗi kết nối LLM: {str(e)}"