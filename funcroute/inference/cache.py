"""
Caching for routing predictions
"""

from typing import Optional, Dict
from collections import OrderedDict
import time
import threading


class RouteCache:
    """
    LRU cache with TTL (Time-To-Live) support for routing results.

    Features:
    - LRU (Least Recently Used) eviction
    - TTL (Time-To-Live) expiration
    - Thread-safe operations
    - Cache statistics

    Example:
        >>> from funcroute.inference import RouteCache
        >>> cache = RouteCache(max_size=1000, ttl_seconds=300)
        >>> cache.put("query", result)
        >>> cached = cache.get("query")
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = None):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries
            ttl_seconds: Time-to-live in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        self._cache = OrderedDict()  # LRU cache
        self._timestamps = {}  # Track insertion times for TTL
        self._lock = threading.RLock()  # Thread-safe

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def get(self, query: str):
        """
        Get cached result for query.

        Args:
            query: Query string

        Returns:
            RouteResult or None if not cached/expired
        """
        with self._lock:
            # Check if exists
            if query not in self._cache:
                self._misses += 1
                return None

            # Check TTL
            if self.ttl_seconds is not None:
                age = time.time() - self._timestamps[query]
                if age > self.ttl_seconds:
                    # Expired - remove
                    del self._cache[query]
                    del self._timestamps[query]
                    self._expirations += 1
                    self._misses += 1
                    return None

            # Cache hit - move to end (most recently used)
            self._cache.move_to_end(query)
            self._hits += 1
            return self._cache[query]

    def put(self, query: str, result):
        """
        Store result in cache.

        Args:
            query: Query string
            result: RouteResult to cache
        """
        with self._lock:
            # Update if exists
            if query in self._cache:
                self._cache.move_to_end(query)
                self._cache[query] = result
                self._timestamps[query] = time.time()
                return

            # Check size limit
            if len(self._cache) >= self.max_size:
                # Evict least recently used (first item)
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]
                del self._timestamps[evicted_key]
                self._evictions += 1

            # Add new entry
            self._cache[query] = result
            self._timestamps[query] = time.time()

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def remove(self, query: str) -> bool:
        """
        Remove specific entry from cache.

        Args:
            query: Query string

        Returns:
            True if removed, False if not in cache
        """
        with self._lock:
            if query in self._cache:
                del self._cache[query]
                del self._timestamps[query]
                return True
            return False

    def get_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "total_requests": total_requests,
            }

    def reset_stats(self):
        """Reset statistics counters."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0

    def cleanup_expired(self) -> int:
        """
        Manually cleanup expired entries.

        Returns:
            Number of entries removed
        """
        if self.ttl_seconds is None:
            return 0

        with self._lock:
            current_time = time.time()
            expired = []

            for query, timestamp in self._timestamps.items():
                age = current_time - timestamp
                if age > self.ttl_seconds:
                    expired.append(query)

            for query in expired:
                del self._cache[query]
                del self._timestamps[query]
                self._expirations += 1

            return len(expired)

    def resize(self, new_max_size: int):
        """
        Resize cache to new maximum size.

        Args:
            new_max_size: New maximum size
        """
        with self._lock:
            self.max_size = new_max_size

            # Evict entries if over new limit
            while len(self._cache) > self.max_size:
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]
                del self._timestamps[evicted_key]
                self._evictions += 1

    def __len__(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, query: str) -> bool:
        """Check if query is in cache (doesn't update LRU or check TTL)."""
        with self._lock:
            return query in self._cache

    def keys(self):
        """Get all cached query keys."""
        with self._lock:
            return list(self._cache.keys())


class WarmupCache(RouteCache):
    """
    Cache with warmup support for common queries.

    Example:
        >>> cache = WarmupCache(max_size=1000)
        >>> cache.warmup(router, common_queries)
    """

    def warmup(self, router, queries: list, show_progress: bool = True):
        """
        Pre-populate cache with common queries.

        Args:
            router: FuncRoute instance
            queries: List of common queries to pre-cache
            show_progress: Show progress bar
        """
        from tqdm import tqdm

        iterator = tqdm(queries, desc="Warming cache") if show_progress else queries

        for query in iterator:
            if query not in self._cache:
                result = router.route(query)
                self.put(query, result)

        if show_progress:
            stats = self.get_stats()
            print(f"\n✅ Cache warmed: {stats['size']} entries")
