"""Tests for RouteCache"""

import pytest
import time
from funcroute.inference.cache import RouteCache, WarmupCache
from funcroute.core.config import RouteResult


def create_mock_result(query: str, tool: str = "tool1") -> RouteResult:
    """Create mock RouteResult"""
    return RouteResult(
        query=query,
        tool=tool,
        confidence=0.9,
        latency_ms=10.0,
    )


def test_cache_init():
    """Test cache initialization"""
    cache = RouteCache(max_size=100, ttl_seconds=60)
    assert cache.max_size == 100
    assert cache.ttl_seconds == 60
    assert len(cache) == 0


def test_cache_put_and_get():
    """Test basic put and get"""
    cache = RouteCache(max_size=10)
    result = create_mock_result("test query")

    cache.put("test query", result)
    cached = cache.get("test query")

    assert cached is not None
    assert cached.query == "test query"
    assert cached.tool == "tool1"


def test_cache_miss():
    """Test cache miss"""
    cache = RouteCache(max_size=10)

    cached = cache.get("nonexistent")
    assert cached is None


def test_cache_lru_eviction():
    """Test LRU eviction when cache is full"""
    cache = RouteCache(max_size=3)

    # Add 3 items
    cache.put("q1", create_mock_result("q1"))
    cache.put("q2", create_mock_result("q2"))
    cache.put("q3", create_mock_result("q3"))

    assert len(cache) == 3

    # Add 4th item - should evict q1 (oldest)
    cache.put("q4", create_mock_result("q4"))

    assert len(cache) == 3
    assert cache.get("q1") is None  # Evicted
    assert cache.get("q2") is not None
    assert cache.get("q3") is not None
    assert cache.get("q4") is not None


def test_cache_lru_update():
    """Test that accessing an item updates its recency"""
    cache = RouteCache(max_size=3)

    cache.put("q1", create_mock_result("q1"))
    cache.put("q2", create_mock_result("q2"))
    cache.put("q3", create_mock_result("q3"))

    # Access q1 - makes it most recent
    cache.get("q1")

    # Add q4 - should evict q2 (now oldest)
    cache.put("q4", create_mock_result("q4"))

    assert cache.get("q1") is not None  # Still in cache
    assert cache.get("q2") is None  # Evicted
    assert cache.get("q3") is not None
    assert cache.get("q4") is not None


def test_cache_ttl_expiration():
    """Test TTL expiration"""
    cache = RouteCache(max_size=10, ttl_seconds=0.1)  # 100ms TTL

    cache.put("q1", create_mock_result("q1"))

    # Immediate get should work
    assert cache.get("q1") is not None

    # Wait for expiration
    time.sleep(0.15)

    # Should be expired
    assert cache.get("q1") is None


def test_cache_no_ttl():
    """Test cache without TTL doesn't expire"""
    cache = RouteCache(max_size=10, ttl_seconds=None)

    cache.put("q1", create_mock_result("q1"))
    time.sleep(0.1)

    # Should still be there
    assert cache.get("q1") is not None


def test_cache_update_existing():
    """Test updating an existing entry"""
    cache = RouteCache(max_size=10)

    result1 = create_mock_result("q1", tool="tool1")
    result2 = create_mock_result("q1", tool="tool2")

    cache.put("q1", result1)
    cache.put("q1", result2)

    cached = cache.get("q1")
    assert cached.tool == "tool2"  # Updated


def test_cache_clear():
    """Test clearing cache"""
    cache = RouteCache(max_size=10)

    cache.put("q1", create_mock_result("q1"))
    cache.put("q2", create_mock_result("q2"))

    assert len(cache) == 2

    cache.clear()

    assert len(cache) == 0
    assert cache.get("q1") is None


def test_cache_remove():
    """Test removing specific entry"""
    cache = RouteCache(max_size=10)

    cache.put("q1", create_mock_result("q1"))
    cache.put("q2", create_mock_result("q2"))

    removed = cache.remove("q1")
    assert removed is True
    assert cache.get("q1") is None
    assert cache.get("q2") is not None

    # Remove non-existent
    removed = cache.remove("q3")
    assert removed is False


def test_cache_stats():
    """Test cache statistics"""
    cache = RouteCache(max_size=10)

    # No requests yet
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["hit_rate"] == 0.0

    # Add and access
    cache.put("q1", create_mock_result("q1"))
    cache.get("q1")  # Hit
    cache.get("q2")  # Miss

    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["size"] == 1


def test_cache_stats_evictions():
    """Test eviction counting in stats"""
    cache = RouteCache(max_size=2)

    cache.put("q1", create_mock_result("q1"))
    cache.put("q2", create_mock_result("q2"))
    cache.put("q3", create_mock_result("q3"))  # Evicts q1

    stats = cache.get_stats()
    assert stats["evictions"] == 1


def test_cache_stats_expirations():
    """Test expiration counting in stats"""
    cache = RouteCache(max_size=10, ttl_seconds=0.1)

    cache.put("q1", create_mock_result("q1"))
    time.sleep(0.15)
    cache.get("q1")  # Should be expired

    stats = cache.get_stats()
    assert stats["expirations"] == 1


def test_cache_reset_stats():
    """Test resetting statistics"""
    cache = RouteCache(max_size=10)

    cache.put("q1", create_mock_result("q1"))
    cache.get("q1")
    cache.get("q2")

    stats = cache.get_stats()
    assert stats["hits"] > 0 or stats["misses"] > 0

    cache.reset_stats()

    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_cache_cleanup_expired():
    """Test manual cleanup of expired entries"""
    cache = RouteCache(max_size=10, ttl_seconds=0.1)

    cache.put("q1", create_mock_result("q1"))
    cache.put("q2", create_mock_result("q2"))

    time.sleep(0.15)

    # Manual cleanup
    removed = cache.cleanup_expired()

    assert removed == 2
    assert len(cache) == 0


def test_cache_resize():
    """Test resizing cache"""
    cache = RouteCache(max_size=5)

    for i in range(5):
        cache.put(f"q{i}", create_mock_result(f"q{i}"))

    assert len(cache) == 5

    # Resize down to 3
    cache.resize(3)

    assert cache.max_size == 3
    assert len(cache) == 3


def test_cache_contains():
    """Test __contains__ method"""
    cache = RouteCache(max_size=10)

    cache.put("q1", create_mock_result("q1"))

    assert "q1" in cache
    assert "q2" not in cache


def test_cache_keys():
    """Test getting cache keys"""
    cache = RouteCache(max_size=10)

    cache.put("q1", create_mock_result("q1"))
    cache.put("q2", create_mock_result("q2"))

    keys = cache.keys()
    assert "q1" in keys
    assert "q2" in keys
    assert len(keys) == 2


def test_cache_thread_safety():
    """Test basic thread safety (no errors)"""
    import threading

    cache = RouteCache(max_size=100)

    def worker():
        for i in range(10):
            cache.put(f"q{i}", create_mock_result(f"q{i}"))
            cache.get(f"q{i}")

    threads = [threading.Thread(target=worker) for _ in range(5)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Should complete without errors
    assert len(cache) > 0
