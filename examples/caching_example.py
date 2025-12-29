"""
Caching Example

Demonstrates RouteCache usage for improving prediction performance through caching.
Shows LRU eviction, TTL expiration, statistics tracking, and performance impact.
"""

from funcroute import FuncRoute
from funcroute.inference import Predictor, RouteCache, WarmupCache
import time
from collections import defaultdict


def example_1_basic_cache():
    """
    Example 1: Basic cache usage

    Show cache hits and misses with simple queries.
    """
    print("=" * 80)
    print("Example 1: Basic Cache Usage")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    cache = RouteCache(max_size=100)
    predictor = Predictor(router, cache=cache)

    queries = [
        "Where is my order?",
        "Find laptops",
        "Where is my order?",  # Duplicate - should hit cache
        "Return this item",
        "Find laptops",  # Duplicate - should hit cache
        "Track package",
    ]

    print(f"\nMaking {len(queries)} predictions (with duplicates)...\n")
    print(f"{'#':<4s} {'Query':<25s} {'Tool':<20s} {'Cached?':<10s}")
    print("-" * 80)

    for i, query in enumerate(queries, 1):
        # Check if cached before prediction
        was_cached = query in cache

        result = predictor._predict_single(query)

        print(
            f"{i:<4d} {query:<25s} {result.tool:<20s} "
            f"{'HIT' if was_cached else 'MISS':<10s}"
        )

    stats = cache.get_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"   Hits: {stats['hits']}")
    print(f"   Misses: {stats['misses']}")
    print(f"   Hit Rate: {stats['hit_rate']:.1%}")
    print(f"   Cache Size: {stats['size']}/{stats['max_size']}")


def example_2_cache_performance():
    """
    Example 2: Performance impact of caching

    Compare prediction times with and without cache.
    """
    print("\n" + "=" * 80)
    print("Example 2: Cache Performance Impact")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")

    # Create repeating queries to show cache benefit
    unique_queries = [
        "Where is my order?",
        "Find laptops",
        "Return this item",
        "Track package",
        "Search for monitors",
    ]

    # Repeat each query multiple times
    queries = unique_queries * 20  # 100 total queries, 5 unique

    # WITHOUT CACHE
    print("\n1️⃣  Without cache:")
    predictor_no_cache = Predictor(router, cache=None)

    start = time.time()
    for query in queries:
        predictor_no_cache._predict_single(query)
    no_cache_time = time.time() - start

    print(f"   Time: {no_cache_time:.2f}s")
    print(f"   Queries/sec: {len(queries) / no_cache_time:.1f}")

    # WITH CACHE
    print("\n2️⃣  With cache:")
    cache = RouteCache(max_size=100)
    predictor_with_cache = Predictor(router, cache=cache)

    start = time.time()
    for query in queries:
        predictor_with_cache._predict_single(query)
    cache_time = time.time() - start

    stats = cache.get_stats()

    print(f"   Time: {cache_time:.2f}s")
    print(f"   Queries/sec: {len(queries) / cache_time:.1f}")
    print(f"   Cache hits: {stats['hits']}")
    print(f"   Cache hit rate: {stats['hit_rate']:.1%}")

    print(f"\n⚡ Speedup: {no_cache_time / cache_time:.1f}x faster with cache!")
    print(f"💡 Cache reduced {stats['hits']} redundant model calls")


def example_3_lru_eviction():
    """
    Example 3: LRU eviction behavior

    Demonstrate how LRU (Least Recently Used) eviction works.
    """
    print("\n" + "=" * 80)
    print("Example 3: LRU Eviction")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    cache = RouteCache(max_size=3)  # Small cache to show eviction
    predictor = Predictor(router, cache=cache)

    print("\n📝 Cache size limit: 3 entries\n")

    queries = ["q1", "q2", "q3", "q4", "q5"]

    for i, query in enumerate(queries, 1):
        print(f"Step {i}: Predicting '{query}'")

        predictor._predict_single(query)

        stats = cache.get_stats()
        keys = list(cache.keys())

        print(f"  Cache contents: {keys}")
        print(f"  Size: {len(cache)}/{cache.max_size}")
        print(f"  Evictions so far: {stats['evictions']}")

        if i == 3:
            print("  💡 Cache is full!")
        elif i > 3:
            print(f"  🗑️  Evicted oldest entry to make room")

        print()

    # Now access an old entry to make it recent
    print("Accessing 'q3' to make it recent...")
    predictor._predict_single("q3")
    print(f"Cache contents: {list(cache.keys())}")

    print("\nAdding 'q6' - should evict 'q4' (least recently used)")
    predictor._predict_single("q6")
    print(f"Cache contents: {list(cache.keys())}")

    print("\n💡 LRU eviction keeps frequently accessed entries in cache")


def example_4_ttl_expiration():
    """
    Example 4: TTL (Time-To-Live) expiration

    Show how cache entries expire after a certain time.
    """
    print("\n" + "=" * 80)
    print("Example 4: TTL Expiration")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    cache = RouteCache(max_size=100, ttl_seconds=2)  # 2 second TTL
    predictor = Predictor(router, cache=cache)

    print("\n📝 Cache TTL: 2 seconds\n")

    # First prediction
    print("1. Making prediction...")
    predictor._predict_single("Where is my order?")
    print(f"   Cache size: {len(cache)}")

    # Immediate re-prediction - should hit cache
    print("\n2. Immediate re-prediction (should hit cache)...")
    start = time.time()
    predictor._predict_single("Where is my order?")
    elapsed = time.time() - start

    stats = cache.get_stats()
    print(f"   Time: {elapsed * 1000:.1f}ms")
    print(f"   Cache hits: {stats['hits']} (HIT!)")

    # Wait for expiration
    print("\n3. Waiting 2.5 seconds for expiration...")
    time.sleep(2.5)

    # Re-prediction after expiration - should miss
    print("\n4. Re-prediction after expiration (should miss cache)...")
    start = time.time()
    predictor._predict_single("Where is my order?")
    elapsed = time.time() - start

    stats = cache.get_stats()
    print(f"   Time: {elapsed * 1000:.1f}ms")
    print(f"   Cache expirations: {stats['expirations']} (EXPIRED!)")
    print(f"   Cache size: {len(cache)} (re-cached)")

    print("\n💡 TTL ensures stale predictions are eventually refreshed")


def example_5_cache_warmup():
    """
    Example 5: Cache warmup

    Pre-populate cache with common queries for better performance.
    """
    print("\n" + "=" * 80)
    print("Example 5: Cache Warmup")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")

    # Common queries to warm up
    common_queries = [
        "Where is my order?",
        "Track my package",
        "Find laptops",
        "Search for monitors",
        "Return this item",
        "Cancel my order",
        "Product details",
        "Check order status",
    ]

    print(f"\n🔥 Warming up cache with {len(common_queries)} common queries...\n")

    # Use WarmupCache for automatic warmup
    cache = WarmupCache(max_size=100, ttl_seconds=300)
    predictor = Predictor(router, cache=cache)

    # Warmup
    start = time.time()
    cache.warmup(predictor, common_queries, show_progress=True)
    warmup_time = time.time() - start

    stats = cache.get_stats()

    print(f"\n✅ Warmup complete in {warmup_time:.2f}s")
    print(f"   Cache size: {stats['size']}")
    print(f"   Ready for fast predictions!")

    # Now make predictions - should all hit cache
    print("\n📊 Testing warmed cache:")

    test_queries = common_queries[:3]
    hits_before = stats["hits"]

    for query in test_queries:
        result = predictor._predict_single(query)
        print(f"   '{query}' → {result.tool} (cached)")

    new_stats = cache.get_stats()
    new_hits = new_stats["hits"] - hits_before

    print(f"\n🎯 All {new_hits} predictions hit the warm cache!")


def example_6_cache_statistics():
    """
    Example 6: Detailed cache statistics

    Monitor cache performance with detailed statistics.
    """
    print("\n" + "=" * 80)
    print("Example 6: Cache Statistics Monitoring")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    cache = RouteCache(max_size=10, ttl_seconds=5)
    predictor = Predictor(router, cache=cache)

    # Simulate workload
    queries = [
        "q1",
        "q2",
        "q3",
        "q1",  # Hit
        "q2",  # Hit
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "q10",
        "q11",  # Evicts q1
        "q12",  # Evicts q2
        "q3",  # Hit (still in cache)
    ]

    print("\n📊 Running workload and tracking statistics...\n")

    def print_stats():
        stats = cache.get_stats()
        print(
            f"   Size: {stats['size']:2d}/{stats['max_size']}  |  "
            f"Hits: {stats['hits']:2d}  |  "
            f"Misses: {stats['misses']:2d}  |  "
            f"Hit Rate: {stats['hit_rate']:5.1%}  |  "
            f"Evictions: {stats['evictions']}"
        )

    for i, query in enumerate(queries, 1):
        predictor._predict_single(query)

        if i % 5 == 0:
            print(f"After {i} queries:")
            print_stats()

    print(f"\nFinal statistics:")
    print_stats()

    # Reset stats
    print("\n🔄 Resetting statistics (cache contents preserved)...")
    cache.reset_stats()

    print("After reset:")
    print_stats()


def example_7_cache_management():
    """
    Example 7: Cache management operations

    Demonstrate cache clearing, resizing, and manual cleanup.
    """
    print("\n" + "=" * 80)
    print("Example 7: Cache Management")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    cache = RouteCache(max_size=10, ttl_seconds=1)
    predictor = Predictor(router, cache=cache)

    # Add some entries
    print("\n1️⃣  Adding 5 entries to cache...")
    for i in range(1, 6):
        predictor._predict_single(f"Query {i}")

    print(f"   Cache size: {len(cache)}")
    print(f"   Keys: {list(cache.keys())}")

    # Manual cleanup of expired
    print("\n2️⃣  Waiting for expiration and cleaning up...")
    time.sleep(1.5)

    removed = cache.cleanup_expired()
    print(f"   Removed {removed} expired entries")
    print(f"   Cache size: {len(cache)}")

    # Add more entries
    print("\n3️⃣  Adding 10 more entries...")
    for i in range(6, 16):
        predictor._predict_single(f"Query {i}")

    print(f"   Cache size: {len(cache)}")

    # Resize cache
    print("\n4️⃣  Resizing cache from 10 to 5...")
    cache.resize(5)

    print(f"   New max size: {cache.max_size}")
    print(f"   Current size: {len(cache)} (evicted excess entries)")
    print(f"   Keys: {list(cache.keys())}")

    # Remove specific entry
    print("\n5️⃣  Removing specific entry...")
    removed = cache.remove("Query 11")

    print(f"   Removed: {removed}")
    print(f"   Cache size: {len(cache)}")

    # Clear all
    print("\n6️⃣  Clearing entire cache...")
    cache.clear()

    print(f"   Cache size: {len(cache)}")
    print(f"   Cache is empty!")


def example_8_production_patterns():
    """
    Example 8: Production caching patterns

    Best practices for using cache in production.
    """
    print("\n" + "=" * 80)
    print("Example 8: Production Caching Patterns")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")

    # Pattern 1: Large cache with TTL
    print("\n1️⃣  Pattern 1: Large cache with TTL (for high-traffic APIs)")
    cache1 = RouteCache(max_size=10000, ttl_seconds=300)  # 5 min TTL
    predictor1 = Predictor(router, cache=cache1)

    print("   ✅ Large cache (10,000 entries)")
    print("   ✅ 5-minute TTL (balance freshness vs performance)")
    print("   📝 Use for: High-traffic APIs with repeating queries")

    # Pattern 2: Medium cache, no TTL
    print("\n2️⃣  Pattern 2: Medium cache, no TTL (for stable models)")
    cache2 = RouteCache(max_size=1000, ttl_seconds=None)
    predictor2 = Predictor(router, cache=cache2)

    print("   ✅ Medium cache (1,000 entries)")
    print("   ✅ No TTL (predictions never expire)")
    print("   📝 Use for: Stable models that don't change often")

    # Pattern 3: Small cache with short TTL
    print("\n3️⃣  Pattern 3: Small cache with short TTL (for frequently updated models)")
    cache3 = RouteCache(max_size=100, ttl_seconds=60)  # 1 min TTL
    predictor3 = Predictor(router, cache=cache3)

    print("   ✅ Small cache (100 entries)")
    print("   ✅ Short 1-minute TTL")
    print("   📝 Use for: Models that update frequently")

    # Pattern 4: Warmup cache for predictable workload
    print("\n4️⃣  Pattern 4: Warmup cache (for predictable workloads)")
    cache4 = WarmupCache(max_size=500, ttl_seconds=600)  # 10 min TTL
    predictor4 = Predictor(router, cache=cache4)

    common_patterns = [
        "Where is my order?",
        "Track package",
        "Find products",
        "Return item",
        "Cancel order",
    ]

    cache4.warmup(predictor4, common_patterns, show_progress=False)

    print("   ✅ Pre-warmed with common queries")
    print("   ✅ 10-minute TTL")
    print("   📝 Use for: Known query patterns (e.g., chatbot FAQs)")

    # Show recommendations
    print("\n💡 Recommendations:")
    print("   - Monitor hit rate (aim for >80% in production)")
    print("   - Adjust cache size based on memory constraints")
    print("   - Use TTL if model updates or data changes")
    print("   - Warm up cache at startup for better initial performance")
    print("   - Reset stats periodically to track current performance")


if __name__ == "__main__":
    import sys
    import os

    if not os.path.exists("./batch_example_model"):
        print("❌ Model not found. Please run batch_prediction_example.py first.")
        sys.exit(1)

    # Run examples
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_1_basic_cache()
        elif example_num == "2":
            example_2_cache_performance()
        elif example_num == "3":
            example_3_lru_eviction()
        elif example_num == "4":
            example_4_ttl_expiration()
        elif example_num == "5":
            example_5_cache_warmup()
        elif example_num == "6":
            example_6_cache_statistics()
        elif example_num == "7":
            example_7_cache_management()
        elif example_num == "8":
            example_8_production_patterns()
        else:
            print("Usage: python caching_example.py [1-8]")
    else:
        # Run all examples
        example_1_basic_cache()
        example_2_cache_performance()
        example_3_lru_eviction()
        example_4_ttl_expiration()
        example_5_cache_warmup()
        example_6_cache_statistics()
        example_7_cache_management()
        example_8_production_patterns()

        print("\n" + "=" * 80)
        print("✅ All caching examples completed!")
        print("=" * 80)
