"""Chat engine with RAG pipeline — retrieves knowledge context then streams LLM responses."""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from source.config import settings
from source.query_engine import _direct_search_sync, _load_parquet, is_domain_ready

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a knowledgeable assistant for the KnowledgeDB system. \
Answer the user's question based on the provided knowledge context.

Rules:
- Base your answers on the provided context when available.
- If the context does not contain enough information, say so honestly.
- Be concise but thorough.
- Use markdown formatting for readability.
- Reference specific entities or concepts from the context when relevant.

{context_section}"""


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ChatSession:
    id: str
    domain: str
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    title: str = "New Chat"


class SessionManager:
    """In-memory store for chat sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def create(self, domain: str) -> ChatSession:
        sid = uuid.uuid4().hex[:8]
        session = ChatSession(id=sid, domain=domain)
        self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> ChatSession | None:
        return self._sessions.get(session_id)

    def list_all(self) -> list[ChatSession]:
        return sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None


sessions = SessionManager()


# ---------------------------------------------------------------------------
# Context retrieval (RAG)
# ---------------------------------------------------------------------------

def _retrieve_context(domain: str, query: str) -> str:
    """Build a textual context block from the indexed knowledge base."""
    if not is_domain_ready(domain):
        return ""

    result = _direct_search_sync(domain, query, top_k=5)
    parts: list[str] = []

    if result.get("entities"):
        parts.append("### Relevant Entities")
        for e in result["entities"]:
            parts.append(f"- **{e['title']}** ({e.get('type', '')}): {e.get('description', '')}")

    if result.get("relationships"):
        parts.append("\n### Relevant Relationships")
        for r in result["relationships"]:
            parts.append(f"- {r['source']} → {r['target']}: {r.get('description', '')}")

    if result.get("sources"):
        parts.append("\n### Relevant Sources")
        for s in result["sources"]:
            title = s.get("title", "")
            body = s.get("summary", "") or s.get("text", "")
            if title:
                parts.append(f"- **{title}**: {body[:500]}")
            else:
                parts.append(f"- {body[:500]}")

    # Fallback: pull raw text_units snippets when structured results are empty
    if not parts:
        text_df = _load_parquet(domain, "text_units")
        if text_df is not None:
            keywords = [w.lower() for w in query.split() if len(w) >= 2]
            snippets = []
            for _, row in text_df.iterrows():
                text = str(row.get("text", ""))
                if any(kw in text.lower() for kw in keywords):
                    snippet = text[:400] + ("..." if len(text) > 400 else "")
                    snippets.append(snippet)
                if len(snippets) >= 3:
                    break
            if snippets:
                parts.append("### Relevant Text Snippets")
                for s in snippets:
                    parts.append(f"- {s}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM streaming
# ---------------------------------------------------------------------------

def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
    )


async def chat_stream(session: ChatSession, user_message: str):
    """Yield streamed tokens from the LLM, augmented with knowledge context."""
    session.messages.append(ChatMessage(role="user", content=user_message))

    if len(session.messages) == 1:
        session.title = user_message[:50] + ("..." if len(user_message) > 50 else "")

    context = await asyncio.to_thread(_retrieve_context, session.domain, user_message)

    context_section = ""
    if context:
        context_section = f"## Knowledge Context (domain: {session.domain})\n\n{context}"

    system_msg = SYSTEM_PROMPT.format(context_section=context_section)

    messages = [{"role": "system", "content": system_msg}]
    for msg in session.messages[-20:]:
        messages.append({"role": msg.role, "content": msg.content})

    client = _get_client()
    full_response: list[str] = []

    try:
        stream = await client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=4096,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response.append(token)
                yield token

        session.messages.append(
            ChatMessage(role="assistant", content="".join(full_response))
        )

    except Exception as exc:
        logger.exception("Chat LLM call failed")
        error_msg = f"Error calling LLM: {exc}"
        session.messages.append(ChatMessage(role="assistant", content=error_msg))
        yield error_msg
