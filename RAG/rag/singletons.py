"""
Singleton instances for DocRAG components.
Reuse across requests to avoid initialization overhead and connection pooling.
"""

from __future__ import annotations

from rag.embedding import GeminiEmbedder
from rag.retriever import Retriever
from rag.context_builder import ContextBuilder
from rag.generator import Generator
from rag.token_manager import TokenManager


from rag.embedding_cache import get_embedding_cache


# Global singleton instances
_embedder_instance: GeminiEmbedder | None = None
_retriever_instance: Retriever | None = None
_context_builder_instance: ContextBuilder | None = None
_generator_instance: Generator | None = None
_token_manager_instance: TokenManager | None = None


def get_embedder() -> GeminiEmbedder:
    """Get or create global GeminiEmbedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = GeminiEmbedder()
    return _embedder_instance


def get_retriever() -> Retriever:
    """Get or create global Retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance


def get_context_builder() -> ContextBuilder:
    """Get or create global ContextBuilder instance."""
    global _context_builder_instance
    if _context_builder_instance is None:
        _context_builder_instance = ContextBuilder(compact_mode=True)
    return _context_builder_instance


def get_generator() -> Generator:
    """Get or create global Generator instance."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = Generator()
    return _generator_instance


def get_token_manager() -> TokenManager:
    """Get or create global TokenManager instance."""
    global _token_manager_instance
    if _token_manager_instance is None:
        _token_manager_instance = TokenManager()
    return _token_manager_instance



def initialize_all() -> dict:
    """
    Initialize all singleton instances at once.
    Call this once at application startup.
    Returns dict of initialized instances.
    """
    get_embedding_cache()  # Initialize cache
    return {
        "embedder": get_embedder(),
        "retriever": get_retriever(),
        "context_builder": get_context_builder(),
        "generator": get_generator(),
        "token_manager": get_token_manager(),
    }


def reset_all() -> None:
    """Reset all singleton instances (for testing/reloading)."""
    global _embedder_instance, _retriever_instance, _context_builder_instance, _generator_instance, _token_manager_instance
    _embedder_instance = None
    _retriever_instance = None
    _context_builder_instance = None
    _generator_instance = None
    _token_manager_instance = None
