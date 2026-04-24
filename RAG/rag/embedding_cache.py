"""
Embedding cache with TTL for reducing redundant API calls.
Caches query embeddings locally to avoid re-embedding identical queries.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional


class EmbeddingCache:
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize embedding cache.
        
        Args:
            ttl_seconds: Time-to-live for cached embeddings in seconds (default: 1 hour).
        """
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}  # key -> {embedding, timestamp}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    @staticmethod
    def _make_key(text: str) -> str:
        """Generate cache key from text (hash for efficiency)."""
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache if exists and not expired.
        
        Args:
            text: Query text to look up
            
        Returns:
            Embedding list if found and valid, None otherwise
        """
        key = self._make_key(text)
        
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        entry = self.cache[key]
        age = time.time() - entry["timestamp"]
        
        if age > self.ttl_seconds:
            # Expired
            del self.cache[key]
            self.stats["evictions"] += 1
            self.stats["misses"] += 1
            return None
        
        self.stats["hits"] += 1
        return entry["embedding"]

    def put(self, text: str, embedding: List[float]) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Query text
            embedding: Vector embedding
        """
        if not text or not embedding:
            return
        
        key = self._make_key(text)
        self.cache[key] = {
            "embedding": embedding,
            "timestamp": time.time(),
        }

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "evictions": self.stats["evictions"],
            "hit_rate": round(hit_rate, 2),
            "cache_size": len(self.cache),
            "total_requests": total_requests,
        }

    def print_stats(self) -> None:
        """Print cache statistics to console."""
        stats = self.get_stats()
        print("\n[Embedding Cache Stats]")
        print(f"  Hits: {stats['hits']}")
        print(f"  Misses: {stats['misses']}")
        print(f"  Evictions: {stats['evictions']}")
        print(f"  Hit Rate: {stats['hit_rate']}%")
        print(f"  Cache Size: {stats['cache_size']} entries")


# Global singleton cache instance
_cache_instance: EmbeddingCache | None = None


def get_embedding_cache() -> EmbeddingCache:
    """Get or create global embedding cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = EmbeddingCache(ttl_seconds=3600)
    return _cache_instance


def reset_embedding_cache() -> None:
    """Reset global embedding cache (for testing)."""
    global _cache_instance
    _cache_instance = None
