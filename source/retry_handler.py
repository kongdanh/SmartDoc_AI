"""
SmartDoc AI — Retry Handler with Exponential Backoff

Handles rate limit (429) errors gracefully with exponential backoff.
Falls back to cached responses when API is unavailable.
"""

import asyncio
import logging
from functools import wraps
from typing import Callable, Optional, Dict, Any, Awaitable

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when API returns 429 or rate limit is exceeded."""
    pass


def exponential_backoff_retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
):
    """
    Decorator for async functions with exponential backoff retry.
    
    Retries with delays: 2s, 4s, 8s (or custom base_delay)
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds (doubles each retry)
        max_delay: Maximum delay between retries
    
    Example:
        @exponential_backoff_retry(max_retries=3, base_delay=2.0)
        async def query_with_retry(...):
            return await call_api(...)
    """
    def decorator(func: Callable[..., Awaitable]) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Retry succeeded on attempt {attempt + 1}")
                    return result
                    
                except RateLimitError as e:
                    last_exception = e
                    
                    if attempt < max_retries - 1:
                        delay = min(
                            base_delay * (2 ** attempt),
                            max_delay
                        )
                        logger.warning(
                            f"Rate limited. Retrying in {delay:.1f}s... "
                            f"(attempt {attempt + 1}/{max_retries}). Error: {str(e)}"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"Failed after {max_retries} retries. "
                            "Using fallback response."
                        )
                        # Return fallback
                        return await _get_fallback_response(*args, **kwargs)
                
                except Exception as e:
                    logger.exception(f"Unexpected error in {func.__name__}")
                    raise
            
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


async def _get_fallback_response(*args, **kwargs) -> Dict[str, Any]:
    """
    Fallback response when API is rate limited or unavailable.
    
    Returns cached data or a user-friendly error message.
    Extracts domain and query from function args/kwargs.
    """
    # Try to extract domain and query from args
    domain = None
    query = None
    
    # Try positional args
    if len(args) > 1:
        domain = args[1]
    if len(args) > 2:
        query = args[2]
    
    # Try kwargs
    if not domain:
        domain = kwargs.get("domain")
    if not query:
        query = kwargs.get("query")
    
    # Try to get cached response
    if domain and query:
        try:
            from source.cache_manager import get_cache_manager
            cache = get_cache_manager()
            
            # Try to get ANY cached response for this query (ignore TTL)
            cached = cache.get(domain, query, method="fallback")
            if cached:
                return {
                    **cached,
                    "is_fallback": True,
                    "message": "⚠️ API rate limited. Showing cached response.",
                }
        except Exception as e:
            logger.warning(f"Failed to get fallback from cache: {e}")
    
    # Final fallback: generic response
    return {
        "response": (
            "Hệ thống đang quá tải. Vui lòng thử lại sau 1-2 phút. "
            "Hãy thử những câu hỏi khác hoặc xem tài liệu gốc."
        ),
        "is_fallback": True,
        "error": "rate_limited",
        "message": "⚠️ API temporarily unavailable. Please try again later.",
    }


# Helper function for checking rate limit errors in responses
def check_rate_limit_error(response: Dict[str, Any], error_str: str = "") -> bool:
    """
    Check if a response or error string indicates rate limiting.
    
    Args:
        response: Response dict to check
        error_str: Error string to check
    
    Returns:
        True if rate limit detected, False otherwise
    """
    # Check response dict
    if response:
        error = response.get("error", "")
        if error and ("429" in str(error) or "rate" in str(error).lower()):
            return True
    
    # Check error string
    if error_str and ("429" in str(error_str) or "rate" in str(error_str).lower()):
        return True
    
    return False
