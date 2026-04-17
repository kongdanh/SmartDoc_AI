"""Convert various file formats to plain text for GraphRAG ingestion."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {".txt", ".text", ".log"}
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".rst"}
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".c", ".h", ".cpp", ".hpp", ".java",
    ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".ps1", ".bat",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".html",
}
CSV_JSON_EXTENSIONS = {".csv", ".json", ".jsonl"}
PDF_EXTENSIONS = {".pdf"}

SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS | MARKDOWN_EXTENSIONS | CODE_EXTENSIONS
    | CSV_JSON_EXTENSIONS | PDF_EXTENSIONS
)


def convert_to_text(file_path: Path) -> str | None:
    """Read a file and return its content as plain text.

    Returns None if the file type is unsupported or conversion fails.
    """
    suffix = file_path.suffix.lower()

    if suffix in TEXT_EXTENSIONS | MARKDOWN_EXTENSIONS | CSV_JSON_EXTENSIONS:
        return _read_text_file(file_path)

    if suffix in CODE_EXTENSIONS:
        return _read_code_file(file_path)

    if suffix in PDF_EXTENSIONS:
        return _read_pdf_file(file_path)

    logger.warning("Unsupported file type: %s", file_path)
    return None


def is_supported(file_path: Path) -> bool:
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


# =========================================================================
# [HÀM MỚI THÊM VÀO] Hàm này phục vụ việc lưu PDF ra TXT cho Standard RAG
# =========================================================================
def convert_pdf_to_txt(pdf_path: Path, output_txt_path: Path) -> Path | None:
    """Đọc file PDF, làm sạch văn bản và xuất ra file TXT tương ứng.
    
    Phục vụ cho luồng lập chỉ mục song song (RAG và GraphRAG).
    """
    text = _read_pdf_file(pdf_path)
    if text:
        try:
            # Lưu nội dung đã làm sạch ra file text
            output_txt_path.write_text(text, encoding="utf-8")
            logger.info("Đã chuyển đổi và lưu %s thành %s", pdf_path.name, output_txt_path.name)
            return output_txt_path
        except Exception:
            logger.exception("Lỗi khi ghi file TXT: %s", output_txt_path)
            return None
    return None


def _read_text_file(file_path: Path) -> str | None:
    try:
        return file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        logger.exception("Failed to read text file: %s", file_path)
        return None


def _read_code_file(file_path: Path) -> str | None:
    """Wrap source code with filename metadata so GraphRAG preserves context."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return (
            f"# Source file: {file_path.name}\n"
            f"# Language: {file_path.suffix.lstrip('.')}\n\n"
            f"{content}"
        )
    except Exception:
        logger.exception("Failed to read code file: %s", file_path)
        return None


def _read_pdf_file(file_path: Path) -> str | None:
    try:
        import fitz  # pymupdf

        doc = fitz.open(str(file_path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        raw = "\n\n".join(pages)
        return _clean_pdf_text(raw)
    except ImportError:
        logger.error("pymupdf not installed — cannot process PDF: %s", file_path)
        return None
    except Exception:
        logger.exception("Failed to read PDF file: %s", file_path)
        return None


def _clean_pdf_text(text: str) -> str:
    """Remove PDF noise to reduce token count for LLM processing."""
    import re

    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if re.match(r"^.{1,60}\s*[–—-]\s*\d+$", stripped):
            continue
        cleaned.append(stripped)

    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result