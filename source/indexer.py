"""Auto-indexing engine: scans data/ folders, preprocesses files, runs GraphRAG indexing."""

import asyncio
import logging
import os
import re
import subprocess
from pathlib import Path
from string import Template

from source.config import settings
from source.file_tracker import FileRegistry
from source.preprocessor import convert_to_text, is_supported

logger = logging.getLogger(__name__)

_indexing_lock = asyncio.Lock()
_indexing_status: dict[str, dict] = {}

GRAPHRAG_WORKFLOW_STEPS = [
    "load_input_documents",
    "create_base_text_units",
    "create_final_documents",
    "extract_graph",
    "extract_graph_nlp",
    "prune_graph",
    "summarize_descriptions",
    "finalize_graph",
    "create_communities",
    "create_final_text_units",
    "create_community_reports",
    "create_community_reports_text",
    "generate_text_embeddings",
]


def get_indexing_status() -> dict[str, dict]:
    return dict(_indexing_status)


def _set_status(domain: str, status: str, detail: str = "", progress: int = 0) -> None:
    _indexing_status[domain] = {"status": status, "detail": detail, "progress": progress}


def discover_domains() -> list[str]:
    """Return names of subdirectories inside data_dir (each is a domain)."""
    data_path = settings.data_path
    if not data_path.exists():
        return []
    return sorted(
        d.name for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def _collect_source_files(domain_data_dir: Path) -> list[Path]:
    """Recursively collect all supported files from a domain's data directory."""
    return sorted(
        f for f in domain_data_dir.rglob("*")
        if f.is_file() and is_supported(f)
    )


def _prepare_domain_dir(domain: str) -> Path:
    """Create the index directory structure for a domain and write settings.yaml."""
    domain_index_dir = settings.index_path / domain
    input_dir = domain_index_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (domain_index_dir / "output").mkdir(exist_ok=True)
    (domain_index_dir / "cache").mkdir(exist_ok=True)

    _write_domain_settings(domain_index_dir)
    return domain_index_dir


def _write_domain_settings(domain_index_dir: Path) -> None:
    """Generate settings.yaml for a domain by expanding the template with env vars."""
    template_path = settings.settings_template_path
    template_text = template_path.read_text(encoding="utf-8")

    env_vars = {
        "LLM_PROVIDER": settings.llm_provider,
        "LLM_BASE_URL": settings.llm_base_url,
        "LLM_MODEL": settings.llm_model,
        "LLM_API_KEY": settings.llm_api_key,
        "LLM_RPM": str(settings.llm_rpm),
        "EMBEDDING_PROVIDER": settings.embedding_provider,
        "EMBEDDING_BASE_URL": settings.embedding_base_url,
        "EMBEDDING_MODEL": settings.embedding_model,
        "EMBEDDING_API_KEY": settings.embedding_api_key,
        "EMBEDDING_RPM": str(settings.embedding_rpm),
        "EMBEDDING_DIMENSION": str(settings.embedding_dimension),
    }
    rendered = Template(template_text).safe_substitute(env_vars)

    settings_path = domain_index_dir / "settings.yaml"
    settings_path.write_text(rendered, encoding="utf-8")


def _preprocess_files(
    source_files: list[Path],
    domain_data_dir: Path,
    input_dir: Path,
) -> int:
    """Convert source files to .txt in the domain's input/ directory.

    Returns the number of files successfully preprocessed.
    """
    count = 0
    for src in source_files:
        text = convert_to_text(src)
        if text is None:
            continue

        rel = src.relative_to(domain_data_dir)
        stem = str(rel).replace(os.sep, "_")
        if not stem.endswith(".txt"):
            stem += ".txt"
        dest = input_dir / stem
        dest.write_text(text, encoding="utf-8")
        count += 1
    return count


async def index_domain(domain: str, registry: FileRegistry, force: bool = False) -> bool:
    """Index a single domain. Returns True on success.

    Acquires a global lock so only one indexing job runs at a time,
    avoiding LLM rate-limit errors from concurrent requests.
    """
    async with _indexing_lock:
        return await _index_domain_unlocked(domain, registry, force)


async def _index_domain_unlocked(domain: str, registry: FileRegistry, force: bool = False) -> bool:
    domain_data_dir = settings.data_path / domain
    if not domain_data_dir.is_dir():
        logger.warning("Domain data directory not found: %s", domain_data_dir)
        _set_status(domain, "error", "data directory not found")
        return False

    source_files = _collect_source_files(domain_data_dir)
    if not source_files:
        logger.info("No supported files in domain '%s', skipping", domain)
        _set_status(domain, "skipped", "no supported files")
        return True

    if not force:
        changed = registry.get_new_or_changed_files(domain, source_files, domain_data_dir)
        if not changed:
            logger.info("Domain '%s' is up-to-date, skipping", domain)
            _set_status(domain, "up_to_date")
            return True
    else:
        registry.clear_domain(domain)

    _set_status(domain, "indexing", f"processing {len(source_files)} files")
    logger.info("Indexing domain '%s' (%d files)...", domain, len(source_files))

    domain_index_dir = _prepare_domain_dir(domain)
    input_dir = domain_index_dir / "input"

    # Clear previous input files
    for old in input_dir.glob("*.txt"):
        old.unlink()

    preprocessed = _preprocess_files(source_files, domain_data_dir, input_dir)
    if preprocessed == 0:
        _set_status(domain, "error", "no files could be preprocessed")
        return False

    success = await _run_graphrag_index(domain_index_dir, domain)

    if success:
        registry.mark_indexed(domain, source_files, domain_data_dir)
        _set_status(domain, "ready")
        logger.info("Domain '%s' indexed successfully", domain)
    else:
        _set_status(domain, "error", "graphrag indexing failed")
        logger.error("Domain '%s' indexing failed", domain)

    return success


def _detect_workflow_step(line: str) -> str | None:
    """Return the workflow step name if the line mentions one."""
    for step in GRAPHRAG_WORKFLOW_STEPS:
        if step in line:
            return step
    return None


def _run_graphrag_index_streaming(domain_index_dir: Path, domain: str) -> int:
    """Run GraphRAG CLI and stream output, updating progress in real-time."""
    proc = subprocess.Popen(
        [
            "graphrag", "index",
            "--root", str(domain_index_dir),
            "--method", settings.indexing_method,
            "--skip-validation",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
    )

    total_steps = len(GRAPHRAG_WORKFLOW_STEPS)
    highest_step = 0
    output_lines: list[str] = []

    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        output_lines.append(line)
        logger.info("GraphRAG: %s", line)

        step = _detect_workflow_step(line)
        if step:
            idx = GRAPHRAG_WORKFLOW_STEPS.index(step) + 1
            if idx > highest_step:
                highest_step = idx
            pct = int(highest_step / total_steps * 100)
            _set_status(domain, "indexing", step, pct)

        pct_match = re.search(r"(\d+)%", line)
        if pct_match and highest_step > 0:
            sub_pct = int(pct_match.group(1))
            base = int((highest_step - 1) / total_steps * 100)
            step_range = int(1 / total_steps * 100)
            combined = min(base + int(step_range * sub_pct / 100), 99)
            _set_status(
                domain, "indexing",
                _indexing_status.get(domain, {}).get("detail", ""),
                combined,
            )

    proc.wait()

    if proc.returncode != 0:
        logger.error(
            "GraphRAG index failed (code %d):\n%s",
            proc.returncode, "\n".join(output_lines[-30:]),
        )
    return proc.returncode


async def _run_graphrag_index(domain_index_dir: Path, domain: str) -> bool:
    """Run GraphRAG indexing via streaming subprocess in a worker thread."""
    try:
        returncode = await asyncio.to_thread(
            _run_graphrag_index_streaming, domain_index_dir, domain,
        )
        return returncode == 0
    except FileNotFoundError:
        logger.error("graphrag CLI not found. Make sure graphrag is installed: pip install graphrag")
        return False
    except Exception:
        logger.exception("Unexpected error during GraphRAG indexing")
        return False


async def scan_and_index_all(force: bool = False) -> dict[str, str]:
    """Scan all domains and index those that need updating.

    Returns a dict of domain -> final status.
    """
    registry = FileRegistry(settings.index_path / "file_registry.json")
    domains = discover_domains()

    if not domains:
        logger.info("No domains found in %s", settings.data_path)
        return {}

    logger.info("Found %d domain(s): %s", len(domains), domains)
    results: dict[str, str] = {}

    for domain in domains:
        ok = await index_domain(domain, registry, force=force)
        results[domain] = "success" if ok else "failed"

    return results
