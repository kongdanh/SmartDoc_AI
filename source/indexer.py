from __future__ import annotations
"""
SmartDoc AI — GraphRAG Indexer.

Discovers domains, runs Microsoft GraphRAG indexing pipeline,
and tracks indexing status.
"""

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from source.config import settings
from source.file_tracker import FileRegistry

logger = logging.getLogger(__name__)

# Global indexing status tracker
_indexing_status: Dict[str, str] = {}


def discover_domains() -> List[str]:
    """Return sorted list of domain names (subdirectories of data_path)."""
    if not settings.data_path.exists():
        return []
    return sorted(
        d.name for d in settings.data_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def get_indexing_status() -> Dict[str, str]:
    """Return current indexing status for all domains."""
    return dict(_indexing_status)


def _prepare_graphrag_input(domain: str) -> Optional[Path]:
    """
    Prepare the input directory for GraphRAG indexing.
    Copies/converts all domain files into the GraphRAG input format (text files).
    
    Returns path to the input directory, or None if no content found.
    """
    domain_data_dir = settings.data_path / domain
    if not domain_data_dir.is_dir():
        return None

    # GraphRAG expects: indexes/<domain>/input/*.txt
    domain_index_dir = settings.index_path / domain
    input_dir = domain_index_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    found_content = False

    for f in domain_data_dir.rglob("*"):
        if not f.is_file():
            continue

        suffix = f.suffix.lower()
        target_txt = input_dir / f"{f.stem}.txt"

        if suffix == ".txt":
            # Copy text files directly
            shutil.copy2(f, target_txt)
            found_content = True

        elif suffix == ".pdf":
            from source.preprocessor import convert_pdf_to_txt
            result = convert_pdf_to_txt(f, target_txt)
            if result:
                found_content = True

        elif suffix in (".docx", ".doc"):
            from source.preprocessor import convert_docx_to_txt
            result = convert_docx_to_txt(f, target_txt)
            if result:
                found_content = True

    return input_dir if found_content else None


def _create_settings_yaml(domain: str) -> Path:
    """
    Create a settings.yaml for GraphRAG indexing from the template.
    Substitutes environment variables into the template.
    """
    from source.config import settings as cfg

    domain_index_dir = cfg.index_path / domain
    domain_index_dir.mkdir(parents=True, exist_ok=True)

    settings_path = domain_index_dir / "settings.yaml"

    # Read template
    template_path = Path(__file__).resolve().parent.parent / "settings_template.yaml"
    if template_path.exists():
        template = template_path.read_text(encoding="utf-8")
    else:
        # Minimal fallback template
        template = _get_fallback_settings_yaml()

    # Substitute env vars
    substitutions = {
        "${LLM_MODEL}": cfg.llm_model,
        "${LLM_API_KEY}": cfg.llm_api_key,
        "${LLM_RPM}": str(cfg.llm_rpm),
        "${EMBEDDING_MODEL}": cfg.embedding_model,
        "${EMBEDDING_DIMENSION}": str(cfg.embedding_dimension),
    }

    for placeholder, value in substitutions.items():
        template = template.replace(placeholder, value)

    settings_path.write_text(template, encoding="utf-8")
    logger.info("Created GraphRAG settings for domain '%s'", domain)
    return settings_path


def _get_fallback_settings_yaml() -> str:
    """Fallback settings.yaml if template is missing."""
    return """
completion_models:
  default_completion_model:
    type: litellm
    model_provider: openrouter
    model: "openrouter/${LLM_MODEL}"
    api_key: "${LLM_API_KEY}"
    model_supports_json: false
    temperature: 0.1
    retry:
      type: exponential_backoff
      max_retries: 5
      base_delay: 3.0
      max_delay: 60.0
    rate_limit:
      type: sliding_window
      period_in_seconds: 60
      requests_per_period: ${LLM_RPM}

embedding_models:
  default_embedding_model:
    type: openrouter
    model_provider: huggingface
    model: "huggingface/${EMBEDDING_MODEL}"
    retry:
      type: exponential_backoff
      max_retries: 10
      base_delay: 5.0
      jitter: true
      max_delay: 60.0

input:
  type: text
  storage:
    type: file
    base_dir: input
    encoding: utf-8

output:
  type: file
  base_dir: output

cache:
  type: json
  storage:
    type: file
    base_dir: cache

chunking:
  type: tokens
  size: 800
  overlap: 100

extract_graph:
  model: default_completion_model
  max_gleanings: 2
  strategy:
    type: graph_intelligence
    model_supports_json: true
  entity_types: [person, organization, location, event, concept, date, money, object]

summarize_descriptions:
  model: default_completion_model
  max_length: 500

community_reports:
  model: default_completion_model
  max_length: 1500
  max_input_length: 8000

parallelization:
  num_threads: 1

vector_store:
  type: lancedb
  db_uri: output/lancedb
  index_schema:
    text_unit_text:
      vector_size: ${EMBEDDING_DIMENSION}
    entity_description:
      vector_size: ${EMBEDDING_DIMENSION}
    community_full_content:
      vector_size: ${EMBEDDING_DIMENSION}
""".strip()


async def index_domain(
    domain: str,
    registry: Optional[FileRegistry] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Run GraphRAG indexing for a single domain.

    Args:
        domain: Name of the domain (subfolder in data_path).
        registry: File registry to track indexed files.
        force: If True, re-index even if already indexed.

    Returns:
        Dict with status and details.
    """
    global _indexing_status

    _indexing_status[domain] = "preparing"
    logger.info("Indexing domain '%s'...", domain)

    try:
        # 1. Prepare input files
        input_dir = _prepare_graphrag_input(domain)
        if not input_dir:
            _indexing_status[domain] = "no_content"
            return {"domain": domain, "status": "error", "message": "No content found"}

        # 2. Create settings.yaml
        _create_settings_yaml(domain)

        # 3. Run graphrag index CLI
        domain_index_dir = settings.index_path / domain
        _indexing_status[domain] = "indexing"

        # Use subprocess to run graphrag CLI
        cmd = [
            sys.executable, "-m", "graphrag", "index",
            "--root", str(domain_index_dir),
        ]

        logger.info("Running: %s", " ".join(cmd))

        def run_indexer():
            return subprocess.run(
                cmd,
                capture_output=True,
                cwd=str(domain_index_dir)
            )

        process = await asyncio.to_thread(run_indexer)

        stdout = process.stdout
        stderr = process.stderr

        if process.returncode == 0:
            _indexing_status[domain] = "ready"
            logger.info("GraphRAG indexing complete for '%s'", domain)

            # Update registry
            if registry:
                domain_data_dir = settings.data_path / domain
                for f in domain_data_dir.rglob("*"):
                    if f.is_file():
                        checksum = hashlib.md5(f.read_bytes()).hexdigest()
                        registry.mark_indexed(domain, f.name, checksum)

            return {
                "domain": domain,
                "status": "success",
                "message": f"Indexing complete for '{domain}'",
            }
        else:
            error_msg = stderr.decode("utf-8", errors="replace")[-500:]
            _indexing_status[domain] = "error"
            logger.error("GraphRAG indexing failed for '%s': %s", domain, error_msg)
            return {
                "domain": domain,
                "status": "error",
                "message": f"Indexing failed: {error_msg}",
            }

    except Exception as e:
        _indexing_status[domain] = "error"
        logger.exception("Indexing error for domain '%s'", domain)
        return {"domain": domain, "status": "error", "message": str(e)}


async def scan_and_index_all(force: bool = False) -> List[Dict[str, Any]]:
    """Scan data directory and index all domains that need indexing."""
    domains = discover_domains()
    if not domains:
        logger.info("No domains found in %s", settings.data_path)
        return []

    registry = FileRegistry(settings.index_path / "file_registry.json")
    results = []

    for domain in domains:
        if not force:
            output_dir = settings.index_path / domain / "output"
            if output_dir.exists() and any(output_dir.iterdir()):
                _indexing_status[domain] = "ready"
                logger.info("Domain '%s' already indexed, skipping", domain)
                continue

        result = await index_domain(domain, registry, force=force)
        results.append(result)

    return results
