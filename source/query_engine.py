"""
SmartDoc AI — GraphRAG Query Engine.

Provides query interfaces for local, global, drift, and direct search.
Wraps the Microsoft GraphRAG library.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from graphrag.config.load_config import load_config
from graphrag.api import local_search, global_search, drift_search

from source.config import settings

logger = logging.getLogger(__name__)

_engine_cache = {}

def _load_domain_data(domain: str) -> dict:
    """Load GraphRAG configuration and dataframes into memory."""
    if domain in _engine_cache:
        return _engine_cache[domain]
        
    domain_root = _get_domain_root(domain)
    output_dir = domain_root / "output" / "artifacts"
    if not output_dir.exists():
        output_dir = domain_root / "output"
        
    config = load_config(domain_root)
    
    def _read_pq(name):
        p = output_dir / f"{name}.parquet"
        return pd.read_parquet(p) if p.exists() else None
        
    data = {
        "config": config,
        "entities": _read_pq("entities"),
        "communities": _read_pq("communities"),
        "community_reports": _read_pq("community_reports"),
        "text_units": _read_pq("text_units"),
        "relationships": _read_pq("relationships"),
    }
    _engine_cache[domain] = data
    return data


def get_available_domains() -> List[str]:
    """Return list of domains that have been indexed (have output dir)."""
    if not settings.index_path.exists():
        return []

    domains = []
    for d in sorted(settings.index_path.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            output_dir = d / "output"
            if output_dir.exists() and any(output_dir.iterdir()):
                domains.append(d.name)
    return domains


def is_domain_ready(domain: str) -> bool:
    """Check if a domain has completed indexing and is ready for queries."""
    output_dir = settings.index_path / domain / "output"
    if not output_dir.exists():
        return False
    # Check for artifacts directory (GraphRAG 3.x output structure)
    artifacts = output_dir / "artifacts"
    if artifacts.exists() and any(artifacts.iterdir()):
        return True
    # Also check for parquet files directly in output
    if any(output_dir.glob("*.parquet")):
        return True
    return False


def _get_domain_root(domain: str) -> Path:
    """Get the root directory for a domain's index."""
    return settings.index_path / domain


async def _run_graphrag_query(
    domain: str,
    query: str,
    method: str = "local",
    community_level: int = 2,
    response_type: str = "Direct Answer",
    **kwargs,
) -> Dict[str, Any]:
    """
    Run a GraphRAG query using the native Python API with in-memory cache.
    """
    if not is_domain_ready(domain):
        return {
            "response": "",
            "method": method,
            "error": f"Domain '{domain}' is not ready. Please index it first.",
        }

    try:
        # Load data in a thread to not block the event loop with pandas I/O
        data = await asyncio.to_thread(_load_domain_data, domain)
        
        if method == "local":
            result, context = await local_search(
                config=data["config"],
                entities=data["entities"],
                communities=data["communities"],
                community_reports=data["community_reports"],
                text_units=data["text_units"],
                relationships=data["relationships"],
                covariates=None,
                community_level=community_level,
                response_type=response_type,
                query=query,
            )
        elif method == "global":
            result, context = await global_search(
                config=data["config"],
                entities=data["entities"],
                communities=data["communities"],
                community_reports=data["community_reports"],
                community_level=community_level,
                dynamic_community_selection=False,
                response_type=response_type,
                query=query,
            )
        elif method == "drift":
            result, context = await drift_search(
                config=data["config"],
                entities=data["entities"],
                communities=data["communities"],
                community_reports=data["community_reports"],
                text_units=data["text_units"],
                relationships=data["relationships"],
                community_level=community_level,
                response_type=response_type,
                query=query,
            )
        else:
            return {"response": "", "method": method, "error": f"Unknown method: {method}"}
            
        return {"response": str(result), "method": method}
        
    except Exception as e:
        logger.exception("GraphRAG Native API error")
        return {"response": "", "method": method, "error": str(e)}


# ── Public Query Functions ──────────────────────────────────────


async def query_local(
    domain: str,
    query: str,
    community_level: int = 2,
    response_type: str = "Direct Answer",
) -> Dict[str, Any]:
    """Local search — entity-focused, specific questions."""
    return await _run_graphrag_query(
        domain, query, method="local",
        community_level=community_level, response_type=response_type,
    )


async def query_global(
    domain: str,
    query: str,
    community_level: int = 2,
    response_type: str = "Direct Answer",
    dynamic_community_selection: bool = False,
) -> Dict[str, Any]:
    """Global search — holistic, dataset-wide questions."""
    return await _run_graphrag_query(
        domain, query, method="global",
        community_level=community_level, response_type=response_type,
    )


async def query_drift(
    domain: str,
    query: str,
    community_level: int = 2,
    response_type: str = "Direct Answer",
) -> Dict[str, Any]:
    """DRIFT search — starts global then drills into specifics."""
    return await _run_graphrag_query(
        domain, query, method="drift",
        community_level=community_level, response_type=response_type,
    )


async def query_auto(
    domain: str,
    query: str,
    community_level: int = 2,
    response_type: str = "Direct Answer",
) -> Dict[str, Any]:
    """Auto-select query method based on query characteristics."""
    query_lower = query.lower()
    broad_indicators = ["tóm tắt", "tổng quan", "tất cả", "toàn bộ", "so sánh", "summary", "overview"]

    if any(indicator in query_lower for indicator in broad_indicators):
        return await query_global(domain, query, community_level, response_type)
    return await query_local(domain, query, community_level, response_type)


async def query_direct(
    domain: str,
    query: str,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Direct search — reads indexed data, returns entities/relationships/sources
    without LLM summarization.
    """
    if not is_domain_ready(domain):
        return {"error": f"Domain '{domain}' is not ready."}

    domain_root = _get_domain_root(domain)

    try:

        output_dir = domain_root / "output"
        artifacts_dir = output_dir / "artifacts"

        entity_file = None
        rel_file = None
        source_file = None

        for search_dir in [artifacts_dir, output_dir]:
            if not search_dir.exists():
                continue
            for f in search_dir.rglob("*.parquet"):
                fname = f.name.lower()
                if "entit" in fname and entity_file is None:
                    entity_file = f
                elif "relat" in fname and rel_file is None:
                    rel_file = f
                elif ("text_unit" in fname or "source" in fname or "document" in fname) and source_file is None:
                    source_file = f

        entities = []
        relationships = []
        sources = []

        if entity_file and entity_file.exists():
            df = pd.read_parquet(entity_file)
            query_lower = query.lower()

            if "title" in df.columns:
                mask = df["title"].str.lower().str.contains(query_lower, na=False)
                if "description" in df.columns:
                    mask = mask | df["description"].str.lower().str.contains(query_lower, na=False)
                filtered = df[mask].head(top_k)

                for _, row in filtered.iterrows():
                    entities.append({
                        "title": str(row.get("title", "")),
                        "type": str(row.get("type", "")),
                        "description": str(row.get("description", ""))[:500],
                    })

        if rel_file and rel_file.exists():
            df = pd.read_parquet(rel_file)
            query_lower = query.lower()

            if "source" in df.columns and "target" in df.columns:
                mask = (
                    df["source"].str.lower().str.contains(query_lower, na=False) |
                    df["target"].str.lower().str.contains(query_lower, na=False)
                )
                if "description" in df.columns:
                    mask = mask | df["description"].str.lower().str.contains(query_lower, na=False)
                filtered = df[mask].head(top_k)

                for _, row in filtered.iterrows():
                    relationships.append({
                        "source": str(row.get("source", "")),
                        "target": str(row.get("target", "")),
                        "description": str(row.get("description", ""))[:300],
                        "weight": float(row.get("weight", 1.0)) if "weight" in row else 1.0,
                    })

        if source_file and source_file.exists():
            df = pd.read_parquet(source_file)
            query_lower = query.lower()

            text_col = "text" if "text" in df.columns else "chunk" if "chunk" in df.columns else None
            if text_col:
                mask = df[text_col].str.lower().str.contains(query_lower, na=False)
                filtered = df[mask].head(top_k)

                for _, row in filtered.iterrows():
                    sources.append({
                        "title": str(row.get("title", row.get("id", "")))[:100],
                        "text": str(row.get(text_col, ""))[:500],
                    })

        return {
            "entities": entities,
            "relationships": relationships,
            "sources": sources,
        }

    except Exception as e:
        logger.exception("Direct search failed for domain '%s'", domain)
        return {"error": str(e)}


async def load_graph(domain: str) -> Dict[str, Any]:
    """
    Load knowledge graph data (nodes + edges) for visualization.
    """
    if not is_domain_ready(domain):
        return {"error": f"Domain '{domain}' is not ready."}

    domain_root = _get_domain_root(domain)

    try:

        output_dir = domain_root / "output"
        artifacts_dir = output_dir / "artifacts"

        entity_file = None
        rel_file = None

        for search_dir in [artifacts_dir, output_dir]:
            if not search_dir.exists():
                continue
            for f in search_dir.rglob("*.parquet"):
                fname = f.name.lower()
                if "entit" in fname and entity_file is None:
                    entity_file = f
                elif "relat" in fname and rel_file is None:
                    rel_file = f

        nodes = []
        edges = []
        text_units = 0

        if entity_file and entity_file.exists():
            df = pd.read_parquet(entity_file)

            degree_map: Dict[str, int] = {}
            if rel_file and rel_file.exists():
                rel_df = pd.read_parquet(rel_file)
                if "source" in rel_df.columns:
                    for s in rel_df["source"]:
                        degree_map[str(s)] = degree_map.get(str(s), 0) + 1
                if "target" in rel_df.columns:
                    for t in rel_df["target"]:
                        degree_map[str(t)] = degree_map.get(str(t), 0) + 1

            for _, row in df.iterrows():
                title = str(row.get("title", row.get("name", row.get("id", ""))))
                entity_type = str(row.get("type", ""))
                description = str(row.get("description", ""))

                nodes.append({
                    "id": title,
                    "type": entity_type,
                    "description": description[:300],
                    "degree": degree_map.get(title, 0),
                })

        if rel_file and rel_file.exists():
            rel_df = pd.read_parquet(rel_file)
            for _, row in rel_df.iterrows():
                source = str(row.get("source", ""))
                target = str(row.get("target", ""))
                if source and target:
                    edges.append({
                        "source": source,
                        "target": target,
                        "description": str(row.get("description", ""))[:200],
                        "weight": float(row.get("weight", 1.0)) if "weight" in row else 1.0,
                    })

        for search_dir in [artifacts_dir, output_dir]:
            if not search_dir.exists():
                continue
            for f in search_dir.rglob("*.parquet"):
                if "text_unit" in f.name.lower():
                    tu_df = pd.read_parquet(f)
                    text_units = len(tu_df)
                    break

        return {
            "nodes": nodes,
            "edges": edges,
            "text_units": text_units,
        }

    except Exception as e:
        logger.exception("Failed to load graph for domain '%s'", domain)
        return {"error": str(e)}