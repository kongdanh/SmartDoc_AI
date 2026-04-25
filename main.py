"""KnowledgeDB — SmartDoc AI FastAPI Service.

Auto-indexes knowledge domains on startup and exposes query endpoints.
Supports both GraphRAG and Standard RAG for comparison.
"""

import sys
import asyncio
from dotenv import load_dotenv
load_dotenv()

# Ép Windows sử dụng ProactorEventLoopPolicy
if sys.platform == 'win32':
    try:
        policy = asyncio.WindowsProactorEventLoopPolicy()
        asyncio.set_event_loop_policy(policy)
    except Exception as e:
        print(f"Lỗi đặt EventLoopPolicy: {e}")

import json
import logging
import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from source.preprocessor import convert_pdf_to_txt, convert_docx_to_txt
from standard_rag import build_standard_rag_index, query_standard_rag, build_faiss_index

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from source.config import settings
from source.dashboard import router as dashboard_router
from source.file_tracker import FileRegistry
from source.indexer import discover_domains, get_indexing_status, scan_and_index_all
from source.chat_engine import chat_stream, sessions
from source.models import (
    ChatRequest,
    CompareRequest,
    DirectSearchRequest,
    DirectSearchResponse,
    DomainInfo,
    DomainListResponse,
    GlobalQueryRequest,
    QueryRequest,
    QueryResponse,
    ReindexResponse,
    StatusResponse,
)
from source.query_engine import (
    get_available_domains,
    is_domain_ready,
    load_graph,
    query_auto,
    query_direct,
    query_drift,
    query_global,
    query_local,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run auto-indexing in background on startup."""
    logger.info("KnowledgeDB starting — scanning for new knowledge domains...")
    logger.info("Data directory: %s", settings.data_path)
    logger.info("Index directory: %s", settings.index_path)
    logger.info("Upload directory: %s", settings.upload_path)

    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.index_path.mkdir(parents=True, exist_ok=True)
    settings.upload_path.mkdir(parents=True, exist_ok=True)

    task = asyncio.create_task(_background_index())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _background_index():
    try:
        results = await scan_and_index_all()
        if results:
            logger.info("Auto-indexing complete: %s", results)
        else:
            logger.info("No domains found to index. Add subfolders to %s", settings.data_path)
    except Exception:
        logger.exception("Background indexing failed")


app = FastAPI(
    title="KnowledgeDB",
    description="SmartDoc AI — GraphRAG & Standard RAG knowledge base API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)


# ─── Health ─────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "SmartDoc AI"}


# --- Domain Endpoints ---


@app.get("/domains", response_model=DomainListResponse)
async def list_domains():
    """List all knowledge domains and their readiness status."""
    data_domains = set(discover_domains())
    indexed_domains = set(get_available_domains())
    all_domains = sorted(data_domains | indexed_domains)

    registry = FileRegistry(settings.index_path / "file_registry.json")
    stats = registry.get_stats()

    items = []
    for name in all_domains:
        items.append(
            DomainInfo(
                name=name,
                ready=is_domain_ready(name),
                indexed_files=stats.get(name, {}).get("indexed_files", 0),
            )
        )

    return DomainListResponse(domains=items)


# --- Query Endpoints ---


@app.post("/query/global", response_model=QueryResponse)
async def search_global(req: GlobalQueryRequest):
    """Global search — holistic, dataset-wide questions."""
    result = await query_global(
        domain=req.domain,
        query=req.query,
        community_level=req.community_level,
        response_type=req.response_type,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return QueryResponse(**result)


@app.post("/query/local", response_model=QueryResponse)
async def search_local(req: QueryRequest):
    """Local search — entity-focused, specific questions."""
    result = await query_local(
        domain=req.domain,
        query=req.query,
        community_level=req.community_level,
        response_type=req.response_type,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return QueryResponse(**result)


@app.post("/query/drift", response_model=QueryResponse)
async def search_drift(req: QueryRequest):
    """DRIFT search — starts global then drills down into details."""
    result = await query_drift(
        domain=req.domain,
        query=req.query,
        community_level=req.community_level,
        response_type=req.response_type,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return QueryResponse(**result)


@app.post("/query/search", response_model=DirectSearchResponse)
async def search_direct(req: DirectSearchRequest):
    """Fast direct search — reads indexed data, no LLM summarisation."""
    result = await query_direct(
        domain=req.domain,
        query=req.query,
        top_k=req.top_k,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return DirectSearchResponse(**result)


@app.get("/graph/{domain}")
async def graph_data(domain: str):
    """Return nodes and edges for knowledge graph visualization."""
    result = await load_graph(domain)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# --- Chat Endpoints ---


@app.post("/chat")
async def chat(req: ChatRequest):
    """Send a chat message and receive a streaming SSE response (RAG-augmented)."""
    if req.session_id:
        session = sessions.get(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
    else:
        session = sessions.create(domain=req.domain)

    async def _event_stream():
        yield f"data: {json.dumps({'type': 'start', 'session_id': session.id})}\n\n"
        async for token in chat_stream(session, req.message):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@app.get("/chat/sessions")
async def list_chat_sessions():
    """List all chat sessions."""
    return {
        "sessions": [
            {
                "id": s.id,
                "domain": s.domain,
                "title": s.title,
                "created_at": s.created_at,
                "message_count": len(s.messages),
            }
            for s in sessions.list_all()
        ]
    }


@app.get("/chat/sessions/{session_id}")
async def get_chat_session(session_id: str):
    """Get a chat session with full message history."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {
        "id": session.id,
        "domain": session.domain,
        "title": session.title,
        "created_at": session.created_at,
        "messages": [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in session.messages
        ],
    }


@app.delete("/chat/sessions")
async def delete_all_chat_sessions():
    """Delete all chat sessions."""
    count = sessions.delete_all()
    return {"message": f"{count} sessions deleted"}


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session."""
    if not sessions.delete(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found")
    return {"message": "Session deleted"}


# --- Management Endpoints ---


@app.get("/status", response_model=StatusResponse)
async def server_status():
    """Server status and indexing progress."""
    all_domains = discover_domains()
    ready = get_available_domains()
    return StatusResponse(
        domains_total=len(all_domains),
        domains_ready=len(ready),
        indexing_status=get_indexing_status(),
    )


@app.post("/reindex/{domain}", response_model=ReindexResponse)
async def reindex_domain(domain: str):
    """Force re-index a specific domain (runs in background)."""
    data_dir = settings.data_path / domain
    if not data_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found in data directory")

    async def _do_reindex():
        registry = FileRegistry(settings.index_path / "file_registry.json")
        from source.indexer import index_domain as _index_domain
        await _index_domain(domain, registry, force=True)

    asyncio.create_task(_do_reindex())

    # Also rebuild Standard RAG index in background
    asyncio.create_task(asyncio.to_thread(build_standard_rag_index, domain))

    return ReindexResponse(message=f"Re-indexing started for domain '{domain}'", domains=[domain])


@app.post("/reindex", response_model=ReindexResponse)
async def reindex_all():
    """Force re-index all domains (runs in background)."""
    domains = discover_domains()
    if not domains:
        raise HTTPException(status_code=404, detail="No domains found in data directory")

    asyncio.create_task(scan_and_index_all(force=True))

    # Also rebuild Standard RAG index for all domains
    for d in domains:
        asyncio.create_task(asyncio.to_thread(build_standard_rag_index, d))

    return ReindexResponse(message="Re-indexing started for all domains", domains=domains)


# --- File Management Endpoints ---


@app.get("/domains/{domain}/files")
async def list_domain_files(domain: str):
    """List source files for a domain."""
    domain_data_dir = settings.data_path / domain
    if not domain_data_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")

    files = []
    for f in sorted(domain_data_dir.rglob("*")):
        if f.is_file():
            files.append({
                "name": str(f.relative_to(domain_data_dir)),
                "size": f.stat().st_size,
            })
    return {"domain": domain, "files": files}


@app.post("/domains/{domain}/upload")
async def upload_files(domain: str, files: list[UploadFile]):
    """Upload files to a domain's data directory and build Standard RAG index."""
    if not re.match(r'^[a-zA-Z0-9_\-]+$', domain):
        raise HTTPException(status_code=400, detail="Domain name must contain only letters, numbers, underscores, or hyphens")
    domain_data_dir = settings.data_path / domain
    domain_data_dir.mkdir(parents=True, exist_ok=True)

    uploaded = 0
    for f in files:
        if not f.filename:
            continue
            
        safe_name = Path(f.filename).name
        suffix = Path(safe_name).suffix.lower()

        # Save to domain data dir
        dest = domain_data_dir / safe_name
        content = await f.read()
        dest.write_bytes(content)

        # Also save to resources/uploads/{type}/ for organized storage
        if suffix == ".pdf":
            upload_type_dir = settings.upload_path / "pdf"
        elif suffix in (".docx", ".doc"):
            upload_type_dir = settings.upload_path / "docx"
        elif suffix == ".txt":
            upload_type_dir = settings.upload_path / "txt"
        else:
            upload_type_dir = settings.upload_path / "txt"

        upload_type_dir.mkdir(parents=True, exist_ok=True)
        (upload_type_dir / safe_name).write_bytes(content)

        # Convert PDF/DOCX to TXT for GraphRAG (needs text input)
        if suffix == ".pdf":
            txt_dest = dest.with_suffix(".txt")
            convert_pdf_to_txt(dest, txt_dest)
        elif suffix in (".docx", ".doc"):
            txt_dest = dest.with_suffix(".txt")
            convert_docx_to_txt(dest, txt_dest)

        uploaded += 1

    # Build Standard RAG index for this domain in background
    if uploaded > 0:
        asyncio.create_task(asyncio.to_thread(build_standard_rag_index, domain))

    return {
        "domain": domain,
        "uploaded": uploaded,
        "message": "Đã lưu file và lập chỉ mục cho Standard RAG. Hãy chạy Reindex cho GraphRAG."
    }


@app.delete("/domains/{domain}/files/{filename:path}")
async def delete_domain_file(domain: str, filename: str):
    """Delete a single file from a domain's data directory."""
    domain_data_dir = settings.data_path / domain
    file_path = domain_data_dir / filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in domain '{domain}'")

    # Safety check: prevent path traversal
    try:
        file_path.resolve().relative_to(domain_data_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")

    file_path.unlink()
    
    # Also delete generated .txt if original was pdf/docx
    if file_path.suffix.lower() in (".pdf", ".docx", ".doc"):
        txt_companion = file_path.with_suffix(".txt")
        if txt_companion.exists():
            txt_companion.unlink()

    return {"message": f"File '{filename}' deleted from domain '{domain}'"}


@app.delete("/domains/{domain}")
async def delete_domain(domain: str):
    """Delete a domain's data and index directories."""
    data_dir = settings.data_path / domain
    index_dir = settings.index_path / domain

    if not data_dir.exists() and not index_dir.exists():
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")

    if data_dir.exists():
        shutil.rmtree(data_dir)
    if index_dir.exists():
        shutil.rmtree(index_dir)

    registry = FileRegistry(settings.index_path / "file_registry.json")
    registry.clear_domain(domain)

    return {"message": f"Domain '{domain}' deleted"}


# --- Compare RAG Endpoint ---



@app.post("/api/compare-rag")
async def compare_rag(req: CompareRequest):
    """API gọi song song cả Standard RAG và GraphRAG để so sánh."""
    
    # Hàm đóng gói gọi GraphRAG Local Search
    async def get_graphrag_answer():
        try:
            res = await query_auto(
                domain=req.domain, 
                query=req.query, 
                community_level=2, 
                response_type="Direct Answer"
            )
            if res.get("error"):
                return f"GraphRAG chưa sẵn sàng cho domain '{req.domain}': {res['error']}"
            return res.get("response", "Không tìm thấy câu trả lời từ GraphRAG.")
        except Exception as e:
            return f"Lỗi GraphRAG: {str(e)}"

    # Hàm đóng gói gọi Standard RAG
    async def get_standard_rag_answer():
        try:
            # Chạy hàm đồng bộ trong threadpool để không block FastAPI
            return await asyncio.to_thread(query_standard_rag, req.query, req.domain)
        except Exception as e:
            return f"Lỗi Standard RAG: {str(e)}"

    # Chạy đua 2 AI cùng lúc (tiết kiệm 50% thời gian)
    graph_res, std_res = await asyncio.gather(
        get_graphrag_answer(),
        get_standard_rag_answer()
    )

    return {
        "question": req.query,
        "standard_rag_answer": std_res,
        "graph_rag_answer": graph_res
    }


# --- Standard RAG Specific Endpoints ---


@app.post("/api/standard-rag/index/{domain}")
async def index_standard_rag(domain: str):
    """Build/rebuild Standard RAG index for a domain."""
    data_dir = settings.data_path / domain
    if not data_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")

    result = await asyncio.to_thread(build_standard_rag_index, domain)
    return result


@app.post("/api/standard-rag/query")
async def standard_rag_query(req: CompareRequest):
    """Query Standard RAG only."""
    answer = await asyncio.to_thread(query_standard_rag, req.query, req.domain)
    return {"question": req.query, "answer": answer, "domain": req.domain}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.server_port,
        reload=True,
        reload_excludes=["indexes/*", "data/*", "*.pyc", "db/*"],
    )