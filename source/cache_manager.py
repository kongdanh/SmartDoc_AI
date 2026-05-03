"""
SmartDoc AI — Smart Cache Manager

Handles:
1. Query normalization (BID == " BID " == "bid?")
2. Response caching (24h TTL)
3. Embedding cache reuse
"""

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import pickle
import logging

logger = logging.getLogger(__name__)


class SmartCacheManager:
    """Intelligent caching layer with query normalization and TTL management."""
    
    def __init__(self, cache_dir: Path = Path("indexes/.cache")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for hot data
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        logger.info(f"Initialized SmartCacheManager with cache dir: {self.cache_dir}")
        
    def normalize_query(self, query: str) -> str:
        """
        Normalize query for caching:
        - Lowercase
        - Remove extra spaces
        - Remove punctuation
        - Deduplicate articles (a, the, an)
        
        Examples:
            "What is BID?" → "what is bid"
            " BID " → "bid"
            "The company" → "company"
        """
        # Lowercase + strip
        q = query.lower().strip()
        
        # Remove common punctuation at end
        q = re.sub(r'[?!,.]$', '', q)
        
        # Collapse multiple spaces
        q = re.sub(r'\s+', ' ', q)
        
        # Remove articles at start (optional)
        q = re.sub(r'^(a|the|an)\s+', '', q)
        
        return q
    
    def get_cache_key(self, domain: str, query: str, method: str = "local") -> str:
        """Generate consistent cache key from domain, query, and method."""
        normalized = self.normalize_query(query)
        key_str = f"{domain}:{method}:{normalized}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def get(self, domain: str, query: str, method: str = "local") -> Optional[Dict]:
        """
        Get cached response if exists and not expired.
        
        Checks memory cache first, then disk cache.
        Returns None if not found or expired.
        """
        cache_key = self.get_cache_key(domain, query, method)
        
        # Check memory cache first (fastest)
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            if not self._is_expired(entry):
                logger.debug(f"Cache HIT (memory): {cache_key}")
                return entry["data"]
            else:
                del self._memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    entry = pickle.load(f)
                    if not self._is_expired(entry):
                        # Restore to memory cache
                        self._memory_cache[cache_key] = entry
                        logger.debug(f"Cache HIT (disk): {cache_key}")
                        return entry["data"]
                    else:
                        cache_file.unlink()  # Delete expired
                        logger.debug(f"Cache EXPIRED: {cache_key}")
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        logger.debug(f"Cache MISS: {cache_key}")
        return None
    
    def set(
        self, 
        domain: str, 
        query: str, 
        response: Dict[str, Any],
        method: str = "local",
        ttl_hours: int = 24
    ) -> None:
        """
        Cache a response with TTL.
        
        Stores in both memory and disk for persistence.
        """
        cache_key = self.get_cache_key(domain, query, method)
        
        entry = {
            "data": response,
            "created_at": datetime.utcnow().isoformat(),
            "ttl_hours": ttl_hours,
        }
        
        # Store in memory
        self._memory_cache[cache_key] = entry
        
        # Store on disk for persistence
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(entry, f)
            logger.debug(f"Cache SET: {cache_key} (TTL: {ttl_hours}h)")
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")
    
    def _is_expired(self, entry: Dict) -> bool:
        """Check if cache entry is expired based on TTL."""
        try:
            created = datetime.fromisoformat(entry["created_at"])
            ttl = timedelta(hours=entry.get("ttl_hours", 24))
            return datetime.utcnow() > (created + ttl)
        except Exception:
            return True  # Default to expired on error
    
    def clear_domain(self, domain: str) -> None:
        """Clear all cache for a specific domain."""
        # Remove from memory
        keys_to_delete = [
            k for k in self._memory_cache.keys()
            if k.startswith(domain)
        ]
        for k in keys_to_delete:
            del self._memory_cache[k]
        
        # Remove from disk
        for f in self.cache_dir.glob(f"{domain}*.pkl"):
            try:
                f.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {f}: {e}")
        
        logger.info(f"Cleared cache for domain: {domain}")
    
    def clear_all(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()
        for f in self.cache_dir.glob("*.pkl"):
            try:
                f.unlink()
            except Exception:
                pass
        logger.info("Cleared all cache")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        disk_entries = len(list(self.cache_dir.glob("*.pkl")))
        return {
            "memory_entries": len(self._memory_cache),
            "disk_entries": disk_entries,
            "cache_dir": str(self.cache_dir),
            "cache_size_mb": sum(
                f.stat().st_size for f in self.cache_dir.glob("*.pkl")
            ) / (1024 * 1024),
        }


# Global instance
_cache_manager = None


def get_cache_manager() -> SmartCacheManager:
    """Get or create the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        from source.config import settings
        _cache_manager = SmartCacheManager(settings.index_path / ".cache")
    return _cache_manager
