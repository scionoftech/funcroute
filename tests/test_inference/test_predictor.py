"""Tests for Predictor"""

import pytest
import asyncio
from funcroute.inference.predictor import Predictor
from funcroute.inference.cache import RouteCache
from funcroute.core.config import RouteResult


class MockRouter:
    """Mock router for testing"""

    def __init__(self):
        self.call_count = 0

    def route(self, query: str) -> RouteResult:
        """Mock routing"""
        self.call_count += 1
        return RouteResult(
            query=query,
            tool=f"tool_for_{query}",
            confidence=0.9,
            latency_ms=10.0,
            alternatives=[("alt1", 0.3), ("alt2", 0.2)],
        )


def test_predictor_init():
    """Test predictor initialization"""
    router = MockRouter()
    predictor = Predictor(router)

    assert predictor.router == router
    assert predictor.cache is None


def test_predictor_init_with_cache():
    """Test predictor with cache"""
    router = MockRouter()
    cache = RouteCache(max_size=10)
    predictor = Predictor(router, cache=cache)

    assert predictor.cache == cache


def test_predict_single():
    """Test single prediction"""
    router = MockRouter()
    predictor = Predictor(router)

    result = predictor._predict_single("test query")

    assert result.query == "test query"
    assert result.tool == "tool_for_test query"
    assert router.call_count == 1


def test_predict_single_with_cache():
    """Test single prediction with caching"""
    router = MockRouter()
    cache = RouteCache(max_size=10)
    predictor = Predictor(router, cache=cache)

    # First call - cache miss
    result1 = predictor._predict_single("test query")
    assert router.call_count == 1

    # Second call - cache hit
    result2 = predictor._predict_single("test query")
    assert router.call_count == 1  # No additional call

    assert result1.tool == result2.tool


def test_predict_batch():
    """Test batch prediction"""
    router = MockRouter()
    predictor = Predictor(router)

    queries = ["q1", "q2", "q3"]
    results = predictor.predict_batch(queries, show_progress=False)

    assert len(results) == 3
    assert all(isinstance(r, RouteResult) for r in results)
    assert results[0].query == "q1"
    assert results[1].query == "q2"
    assert results[2].query == "q3"


def test_predict_batch_empty():
    """Test batch prediction with empty list"""
    router = MockRouter()
    predictor = Predictor(router)

    results = predictor.predict_batch([])

    assert results == []


def test_predict_batch_single():
    """Test batch prediction with single query"""
    router = MockRouter()
    predictor = Predictor(router)

    results = predictor.predict_batch(["q1"], show_progress=False)

    assert len(results) == 1
    assert results[0].query == "q1"


def test_predict_batch_preserves_order():
    """Test that batch prediction preserves order"""
    router = MockRouter()
    predictor = Predictor(router)

    queries = [f"q{i}" for i in range(10)]
    results = predictor.predict_batch(queries, show_progress=False)

    for i, result in enumerate(results):
        assert result.query == f"q{i}"


def test_predict_stream():
    """Test streaming prediction"""
    router = MockRouter()
    predictor = Predictor(router)

    queries = ["q1", "q2", "q3"]
    results = list(predictor.predict_stream(queries))

    assert len(results) == 3
    # Order may not be preserved in streaming
    result_queries = {r.query for r in results}
    assert result_queries == {"q1", "q2", "q3"}


def test_predict_stream_empty():
    """Test streaming with empty list"""
    router = MockRouter()
    predictor = Predictor(router)

    results = list(predictor.predict_stream([]))

    assert results == []


def test_predict_with_callback():
    """Test prediction with callback"""
    router = MockRouter()
    predictor = Predictor(router)

    callback_results = []

    def callback(result):
        callback_results.append(result)

    queries = ["q1", "q2", "q3"]
    predictor.predict_with_callback(queries, callback)

    assert len(callback_results) == 3


def test_predict_with_filter():
    """Test prediction with filtering"""
    router = MockRouter()
    predictor = Predictor(router)

    queries = [f"q{i}" for i in range(10)]

    # Filter to only keep first 3
    filtered = predictor.predict_with_filter(
        queries, lambda r: int(r.query[1]) < 3
    )

    assert len(filtered) == 3
    assert all(int(r.query[1]) < 3 for r in filtered)


def test_predict_until():
    """Test prediction until condition"""
    router = MockRouter()
    predictor = Predictor(router)

    queries = [f"q{i}" for i in range(100)]

    # Stop after 5 results
    results = predictor.predict_until(queries, lambda results: len(results) >= 5)

    assert len(results) == 5


def test_cache_stats():
    """Test getting cache stats"""
    router = MockRouter()
    cache = RouteCache(max_size=10)
    predictor = Predictor(router, cache=cache)

    # No cache
    predictor_no_cache = Predictor(router)
    assert predictor_no_cache.get_cache_stats() is None

    # With cache
    predictor._predict_single("q1")
    predictor._predict_single("q1")  # Hit

    stats = predictor.get_cache_stats()
    assert stats is not None
    assert stats["hits"] == 1


def test_clear_cache():
    """Test clearing cache"""
    router = MockRouter()
    cache = RouteCache(max_size=10)
    predictor = Predictor(router, cache=cache)

    predictor._predict_single("q1")
    assert len(cache) == 1

    predictor.clear_cache()
    assert len(cache) == 0


@pytest.mark.asyncio
async def test_predict_async():
    """Test async prediction"""
    router = MockRouter()
    predictor = Predictor(router)

    result = await predictor.predict_async("test query")

    assert result.query == "test query"
    assert isinstance(result, RouteResult)


@pytest.mark.asyncio
async def test_predict_batch_async():
    """Test async batch prediction"""
    router = MockRouter()
    predictor = Predictor(router)

    queries = ["q1", "q2", "q3"]
    results = await predictor.predict_batch_async(queries)

    assert len(results) == 3
    assert all(isinstance(r, RouteResult) for r in results)


@pytest.mark.asyncio
async def test_predict_batch_async_empty():
    """Test async batch with empty list"""
    router = MockRouter()
    predictor = Predictor(router)

    results = await predictor.predict_batch_async([])

    assert results == []


def test_batch_with_max_workers():
    """Test batch prediction with custom max_workers"""
    router = MockRouter()
    predictor = Predictor(router)

    queries = [f"q{i}" for i in range(10)]
    results = predictor.predict_batch(queries, max_workers=2, show_progress=False)

    assert len(results) == 10


def test_predict_with_cache_disabled():
    """Test prediction with cache disabled"""
    router = MockRouter()
    cache = RouteCache(max_size=10)
    predictor = Predictor(router, cache=cache)

    # Predict with cache disabled
    result1 = predictor._predict_single("q1", use_cache=False)
    result2 = predictor._predict_single("q1", use_cache=False)

    # Both should call router (no caching)
    assert router.call_count == 2


def test_batch_prediction_with_cache():
    """Test batch prediction uses cache"""
    router = MockRouter()
    cache = RouteCache(max_size=100)
    predictor = Predictor(router, cache=cache)

    queries = ["q1", "q2", "q3"]

    # First batch
    results1 = predictor.predict_batch(queries, show_progress=False)
    count_after_first = router.call_count

    # Second batch (should use cache)
    results2 = predictor.predict_batch(queries, show_progress=False)

    # No additional router calls
    assert router.call_count == count_after_first
