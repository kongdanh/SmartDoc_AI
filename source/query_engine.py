"""Query engine wrapping GraphRAG search methods via CLI subprocess."""

import asyncio
import logging
import os
import subprocess
from pathlib import Path

import pandas as pd

from source.config import settings

logger = logging.getLogger(__name__)


def get_available_domains() -> list[str]:
    """Return domains that have completed indexing (have output/ with parquet files)."""
    index_path = settings.index_path
    if not index_path.exists():
        return []
    domains = []
    for d in sorted(index_path.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        output_dir = d / "output"
        if output_dir.exists() and any(output_dir.glob("*.parquet")):
            domains.append(d.name)
    return domains


def is_domain_ready(domain: str) -> bool:
    """Check if a domain has been indexed and is ready for queries."""
    output_dir = settings.index_path / domain / "output"
    if not output_dir.exists():
        return False
    required = ["entities.parquet", "communities.parquet", "community_reports.parquet"]
    return all((output_dir / f).exists() for f in required)


async def query_global(
    domain: str,
    query: str,
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
    dynamic_community_selection: bool = False,
) -> dict:
    """Run a global search query against a domain's index."""
    if not is_domain_ready(domain):
        return {"error": f"Domain '{domain}' is not indexed or not ready"}

    root_dir = str(settings.index_path / domain)
    cmd = [
        "graphrag", "query", query,
        "--root", root_dir,
        "--method", "global",
        "--community-level", str(community_level),
        "--response-type", response_type,
    ]
    if dynamic_community_selection:
        cmd.append("--dynamic-community-selection")

    return await _run_query(cmd, domain, "global")


async def query_local(
    domain: str,
    query: str,
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
) -> dict:
    """Run a local search query against a domain's index."""
    if not is_domain_ready(domain):
        return {"error": f"Domain '{domain}' is not indexed or not ready"}

    root_dir = str(settings.index_path / domain)
    cmd = [
        "graphrag", "query", query,
        "--root", root_dir,
        "--method", "local",
        "--community-level", str(community_level),
        "--response-type", response_type,
    ]

    return await _run_query(cmd, domain, "local")


async def query_drift(
    domain: str,
    query: str,
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
) -> dict:
    """Run a DRIFT search query (hybrid global -> local drill-down)."""
    if not is_domain_ready(domain):
        return {"error": f"Domain '{domain}' is not indexed or not ready"}

    root_dir = str(settings.index_path / domain)
    cmd = [
        "graphrag", "query", query,
        "--root", root_dir,
        "--method", "drift",
        "--community-level", str(community_level),
        "--response-type", response_type,
    ]

    return await _run_query(cmd, domain, "drift")


async def query_auto(
    domain: str,
    query: str,
    community_level: int = 2,
    response_type: str = "Multiple Paragraphs",
) -> dict:
    """Auto-select search method: try local first, fall back to global."""
    result = await query_local(domain, query, community_level, response_type)
    if result.get("error"):
        return result

    if not result.get("response") or len(result.get("response", "")) < 20:
        logger.info("Local search returned weak result, trying global for domain '%s'", domain)
        return await query_global(domain, query, community_level, response_type)

    return result


def _load_parquet(domain: str, table: str) -> pd.DataFrame | None:
    path = settings.index_path / domain / "output" / f"{table}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _text_match(text: str | None, keywords: list[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def _score_text(text: str | None, keywords: list[str]) -> int:
    if not text:
        return 0
    lower = text.lower()
    return sum(lower.count(kw) for kw in keywords)


def _direct_search_sync(domain: str, query: str, top_k: int = 10) -> dict:
    """Search parquet files directly — no LLM call, pure text matching."""
    keywords = [w.lower() for w in query.split() if len(w) >= 2]
    if not keywords:
        return {"error": "Query too short", "domain": domain, "method": "direct"}

    results: dict = {"domain": domain, "method": "direct", "entities": [], "relationships": [], "sources": []}

    entities_df = _load_parquet(domain, "entities")
    if entities_df is not None:
        scored = []
        for _, row in entities_df.iterrows():
            title = str(row.get("title", ""))
            desc = str(row.get("description", ""))
            score = _score_text(title, keywords) * 3 + _score_text(desc, keywords)
            if score > 0:
                scored.append((score, {
                    "title": title,
                    "type": str(row.get("type", "")),
                    "description": desc,
                }))
        scored.sort(key=lambda x: x[0], reverse=True)
        results["entities"] = [item for _, item in scored[:top_k]]

    rels_df = _load_parquet(domain, "relationships")
    if rels_df is not None:
        scored = []
        for _, row in rels_df.iterrows():
            src = str(row.get("source", ""))
            tgt = str(row.get("target", ""))
            desc = str(row.get("description", ""))
            score = (_score_text(src, keywords) + _score_text(tgt, keywords)) * 2 + _score_text(desc, keywords)
            if score > 0:
                scored.append((score, {
                    "source": src,
                    "target": tgt,
                    "description": desc,
                    "weight": float(row.get("weight", 0)),
                }))
        scored.sort(key=lambda x: x[0], reverse=True)
        results["relationships"] = [item for _, item in scored[:top_k]]

    reports_df = _load_parquet(domain, "community_reports")
    if reports_df is not None:
        scored = []
        for _, row in reports_df.iterrows():
            title = str(row.get("title", ""))
            summary = str(row.get("summary", ""))
            score = _score_text(title, keywords) * 3 + _score_text(summary, keywords)
            if score > 0:
                scored.append((score, {
                    "title": title,
                    "summary": summary,
                }))
        scored.sort(key=lambda x: x[0], reverse=True)
        results["sources"] = [item for _, item in scored[:top_k]]

    if not results["entities"] and not results["relationships"] and not results["sources"]:
        text_df = _load_parquet(domain, "text_units")
        if text_df is not None:
            scored = []
            for _, row in text_df.iterrows():
                text = str(row.get("text", ""))
                score = _score_text(text, keywords)
                if score > 0:
                    snippet = text[:500] + "..." if len(text) > 500 else text
                    scored.append((score, {"text": snippet}))
            scored.sort(key=lambda x: x[0], reverse=True)
            results["sources"] = [item for _, item in scored[:top_k]]

    return results


async def query_direct(domain: str, query: str, top_k: int = 10) -> dict:
    """Fast direct search — reads indexed parquet files, no LLM involved."""
    if not is_domain_ready(domain):
        return {"error": f"Domain '{domain}' is not indexed or not ready", "domain": domain, "method": "direct"}
    return await asyncio.to_thread(_direct_search_sync, domain, query, top_k)


def _load_graph_sync(domain: str) -> dict:
    """Load full graph data (nodes + edges) from parquet files."""
    nodes = []
    edges = []
    node_ids = set()

    entities_df = _load_parquet(domain, "entities")
    if entities_df is not None:
        for _, row in entities_df.iterrows():
            title = str(row.get("title", ""))
            nodes.append({
                "id": title,
                "type": str(row.get("type", "")),
                "description": str(row.get("description", ""))[:300],
                "degree": int(row.get("degree", 1)),
            })
            node_ids.add(title)

    rels_df = _load_parquet(domain, "relationships")
    if rels_df is not None:
        for _, row in rels_df.iterrows():
            src = str(row.get("source", ""))
            tgt = str(row.get("target", ""))
            if src not in node_ids:
                nodes.append({"id": src, "type": "IMPLICIT", "description": "", "degree": 1})
                node_ids.add(src)
            if tgt not in node_ids:
                nodes.append({"id": tgt, "type": "IMPLICIT", "description": "", "degree": 1})
                node_ids.add(tgt)
            edges.append({
                "source": src,
                "target": tgt,
                "weight": float(row.get("weight", 1)),
                "description": str(row.get("description", ""))[:200],
            })

    text_df = _load_parquet(domain, "text_units")
    text_count = len(text_df) if text_df is not None else 0

    return {"domain": domain, "nodes": nodes, "edges": edges, "text_units": text_count}


async def load_graph(domain: str) -> dict:
    """Load graph data for visualization."""
    if not is_domain_ready(domain):
        return {"error": f"Domain '{domain}' is not indexed or not ready"}
    return await asyncio.to_thread(_load_graph_sync, domain)


def _run_query_sync(cmd: list[str]) -> subprocess.CompletedProcess:
    """Blocking helper executed in a worker thread."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, capture_output=True, env=env)


async def _run_query(cmd: list[str], domain: str, method: str) -> dict:
    """Execute a GraphRAG query via subprocess and parse the output.

    Uses subprocess.run in a thread to avoid the Windows SelectorEventLoop
    limitation (asyncio.create_subprocess_exec raises NotImplementedError).
    """
    try:
        proc = await asyncio.to_thread(_run_query_sync, cmd)

        stdout_text = proc.stdout.decode(errors="replace").strip()
        stderr_text = proc.stderr.decode(errors="replace").strip()

        if proc.returncode != 0:
            logger.error(
                "GraphRAG query failed (code %d) for domain '%s':\n%s",
                proc.returncode, domain, stderr_text,
            )
            return {
                "error": f"Query failed: {stderr_text[:500]}",
                "domain": domain,
                "method": method,
            }

        return {
            "response": stdout_text,
            "domain": domain,
            "method": method,
        }

    except FileNotFoundError:
        return {"error": "graphrag CLI not found. Make sure graphrag is installed."}
    except Exception as exc:
        logger.exception("Unexpected error during query")
        return {"error": str(exc), "domain": domain, "method": method}
