"""
Async Prediction Example

Demonstrates asynchronous prediction patterns using asyncio.
Useful for integrating FuncRoute into async web frameworks like FastAPI, aiohttp, etc.
"""

from funcroute import FuncRoute
from funcroute.inference import Predictor, RouteCache
import asyncio
import time
from typing import List


def example_1_basic_async():
    """
    Example 1: Basic async prediction

    Simple async prediction with asyncio.run().
    """
    print("=" * 80)
    print("Example 1: Basic Async Prediction")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    async def predict_one():
        result = await predictor.predict_async("Where is my order?")
        return result

    print("\n🔄 Making async prediction...")

    # Run async function
    result = asyncio.run(predict_one())

    print(f"\n📊 Result:")
    print(f"   Query: {result.query}")
    print(f"   Tool: {result.tool}")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Latency: {result.latency_ms:.1f}ms")


def example_2_concurrent_predictions():
    """
    Example 2: Concurrent async predictions

    Run multiple predictions concurrently using asyncio.gather().
    """
    print("\n" + "=" * 80)
    print("Example 2: Concurrent Async Predictions")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    queries = [
        "Where is my order?",
        "Find laptops",
        "Return this item",
        "Track package",
        "Search for monitors",
    ]

    async def predict_concurrent():
        # Create tasks
        tasks = [predictor.predict_async(query) for query in queries]

        # Wait for all to complete concurrently
        start = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        return results, elapsed

    print(f"\n🔄 Running {len(queries)} predictions concurrently...\n")

    results, elapsed = asyncio.run(predict_concurrent())

    print(f"{'Query':<30s} {'Tool':<20s} {'Confidence':>12s}")
    print("-" * 80)

    for result in results:
        print(
            f"{result.query:<30s} {result.tool:<20s} {result.confidence:>11.1%}"
        )

    print(f"\n✅ Completed {len(queries)} predictions in {elapsed:.2f}s")
    print(f"💡 Concurrent execution allows parallel processing")


def example_3_async_batch():
    """
    Example 3: Async batch prediction

    Use predict_batch_async for ordered batch processing.
    """
    print("\n" + "=" * 80)
    print("Example 3: Async Batch Prediction")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    queries = [f"Query {i}: Find item type {i % 3}" for i in range(20)]

    async def batch_predict():
        start = time.time()
        results = await predictor.predict_batch_async(queries)
        elapsed = time.time() - start
        return results, elapsed

    print(f"\n🔄 Batch predicting {len(queries)} queries asynchronously...\n")

    results, elapsed = asyncio.run(batch_predict())

    # Show first 5 results
    print("First 5 results:")
    for i, result in enumerate(results[:5], 1):
        print(f"  {i}. '{result.query}' → {result.tool}")

    print(f"\n✅ Completed {len(results)} predictions in {elapsed:.2f}s")
    print(f"   Average: {elapsed / len(results) * 1000:.1f}ms per query")


def example_4_async_with_cache():
    """
    Example 4: Async predictions with caching

    Combine async predictions with caching for better performance.
    """
    print("\n" + "=" * 80)
    print("Example 4: Async Predictions with Cache")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    cache = RouteCache(max_size=100)
    predictor = Predictor(router, cache=cache)

    # Queries with duplicates
    queries = ["q1", "q2", "q3"] * 5  # 15 total, 3 unique

    async def predict_with_cache():
        results = []

        for query in queries:
            result = await predictor.predict_async(query)
            results.append(result)

        return results

    print(f"\n🔄 Making {len(queries)} async predictions (with duplicates)...\n")

    results = asyncio.run(predict_with_cache())

    stats = cache.get_stats()

    print(f"📊 Cache Performance:")
    print(f"   Total predictions: {len(results)}")
    print(f"   Cache hits: {stats['hits']}")
    print(f"   Cache misses: {stats['misses']}")
    print(f"   Hit rate: {stats['hit_rate']:.1%}")
    print(f"\n💡 Caching works seamlessly with async predictions!")


def example_5_async_web_integration():
    """
    Example 5: Integration with async web framework

    Simulate FastAPI-style async endpoint handlers.
    """
    print("\n" + "=" * 80)
    print("Example 5: Async Web Framework Integration")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    cache = RouteCache(max_size=1000, ttl_seconds=300)
    predictor = Predictor(router, cache=cache)

    # Simulate FastAPI endpoint handler
    async def handle_route_request(query: str):
        """Simulates an async API endpoint"""
        result = await predictor.predict_async(query)

        return {
            "query": result.query,
            "tool": result.tool,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
        }

    async def handle_batch_request(queries: List[str]):
        """Simulates an async batch API endpoint"""
        results = await predictor.predict_batch_async(queries)

        return {
            "total": len(results),
            "results": [
                {
                    "query": r.query,
                    "tool": r.tool,
                    "confidence": r.confidence,
                }
                for r in results
            ],
        }

    # Simulate API calls
    async def simulate_api_calls():
        print("\n📡 Simulating API calls...\n")

        # Single request
        print("1. Single query request:")
        response1 = await handle_route_request("Where is my order?")
        print(f"   Response: {response1['tool']} ({response1['confidence']:.1%})")

        # Batch request
        print("\n2. Batch query request:")
        queries = ["Find laptops", "Return item", "Track package"]
        response2 = await handle_batch_request(queries)
        print(f"   Total results: {response2['total']}")
        for r in response2["results"]:
            print(f"     - {r['query']} → {r['tool']}")

        # Concurrent requests
        print("\n3. Concurrent requests (like multiple users):")
        tasks = [
            handle_route_request("Order status?"),
            handle_route_request("Find products"),
            handle_route_request("Return item"),
        ]
        responses = await asyncio.gather(*tasks)
        print(f"   Handled {len(responses)} concurrent requests")

    asyncio.run(simulate_api_calls())

    print("\n✅ Async integration pattern ready for FastAPI/aiohttp!")


def example_6_async_streaming():
    """
    Example 6: Async streaming with async generators

    Create async generator for streaming results.
    """
    print("\n" + "=" * 80)
    print("Example 6: Async Streaming")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    queries = [f"Query {i}" for i in range(10)]

    async def stream_predictions(queries):
        """Async generator for streaming predictions"""
        for query in queries:
            result = await predictor.predict_async(query)
            yield result

    async def consume_stream():
        print(f"\n🔄 Streaming {len(queries)} predictions...\n")
        print(f"{'#':<4s} {'Query':<15s} {'Tool':<20s}")
        print("-" * 80)

        count = 0
        async for result in stream_predictions(queries):
            count += 1
            print(f"{count:<4d} {result.query:<15s} {result.tool:<20s}")

        print(f"\n✅ Streamed {count} results")

    asyncio.run(consume_stream())


def example_7_async_with_timeout():
    """
    Example 7: Async predictions with timeout

    Handle timeouts and cancellation in async predictions.
    """
    print("\n" + "=" * 80)
    print("Example 7: Async with Timeout")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    async def predict_with_timeout(query: str, timeout_seconds: float):
        """Predict with timeout"""
        try:
            result = await asyncio.wait_for(
                predictor.predict_async(query), timeout=timeout_seconds
            )
            return result, False  # Not timed out
        except asyncio.TimeoutError:
            return None, True  # Timed out

    async def run_with_timeouts():
        print("\n🔄 Testing predictions with timeouts...\n")

        # Normal timeout (should complete)
        print("1. Normal timeout (10 seconds):")
        result, timed_out = await predict_with_timeout(
            "Where is my order?", timeout_seconds=10.0
        )

        if not timed_out:
            print(f"   ✅ Completed: {result.tool}")
        else:
            print("   ❌ Timed out")

        # Very short timeout (might timeout)
        print("\n2. Very short timeout (0.001 seconds):")
        result, timed_out = await predict_with_timeout(
            "Find laptops", timeout_seconds=0.001
        )

        if not timed_out:
            print(f"   ✅ Completed: {result.tool}")
        else:
            print("   ⏱️  Timed out (expected with very short timeout)")

        # Concurrent with different timeouts
        print("\n3. Concurrent with different timeouts:")
        tasks = [
            predict_with_timeout("q1", 10.0),
            predict_with_timeout("q2", 10.0),
            predict_with_timeout("q3", 0.001),  # Likely to timeout
        ]

        results = await asyncio.gather(*tasks)

        completed = sum(1 for _, timed_out in results if not timed_out)
        print(f"   Completed: {completed}/{len(tasks)}")

    asyncio.run(run_with_timeouts())

    print("\n💡 Use timeouts to prevent slow predictions from blocking!")


def example_8_async_event_loop_integration():
    """
    Example 8: Integration with existing event loop

    Use FuncRoute in an application that already has an event loop.
    """
    print("\n" + "=" * 80)
    print("Example 8: Event Loop Integration")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    async def main_application():
        """Simulate main application with event loop"""

        print("\n🔄 Application started with event loop...\n")

        # Simulate other async work
        async def other_async_work():
            print("  🔧 Doing other async work...")
            await asyncio.sleep(0.1)
            print("  ✅ Other work complete")

        # Prediction task
        async def prediction_task():
            print("  🤖 Making prediction...")
            result = await predictor.predict_async("Where is my order?")
            print(f"  ✅ Prediction complete: {result.tool}")

        # Run both concurrently
        print("Running multiple async tasks concurrently:")
        await asyncio.gather(other_async_work(), prediction_task())

        print("\n💡 FuncRoute predictions integrate seamlessly with other async tasks!")

    # Run with asyncio.run (creates new event loop)
    asyncio.run(main_application())

    # Alternative: Use existing event loop (for applications that manage their own loop)
    print("\n" + "-" * 80)
    print("Alternative: Using get_event_loop() for existing loop")
    print("-" * 80)

    async def simple_predict():
        result = await predictor.predict_async("Track package")
        return result

    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    result = loop.run_until_complete(simple_predict())
    print(f"Result: {result.tool}")


def example_9_production_async_patterns():
    """
    Example 9: Production async patterns

    Best practices for async predictions in production.
    """
    print("\n" + "=" * 80)
    print("Example 9: Production Async Patterns")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    cache = RouteCache(max_size=10000, ttl_seconds=300)
    predictor = Predictor(router, cache=cache)

    # Pattern 1: Semaphore for rate limiting
    print("\n1️⃣  Pattern 1: Rate limiting with semaphore")

    async def rate_limited_predictions():
        # Limit concurrent predictions
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent

        async def predict_with_limit(query):
            async with semaphore:
                return await predictor.predict_async(query)

        queries = [f"Query {i}" for i in range(20)]
        tasks = [predict_with_limit(q) for q in queries]

        results = await asyncio.gather(*tasks)
        return results

    results = asyncio.run(rate_limited_predictions())
    print(f"   ✅ Processed {len(results)} queries with rate limiting")
    print("   📝 Use for: Preventing resource exhaustion")

    # Pattern 2: Error handling with gather
    print("\n2️⃣  Pattern 2: Graceful error handling")

    async def error_resilient_predictions():
        async def safe_predict(query):
            try:
                return await predictor.predict_async(query)
            except Exception as e:
                print(f"     ⚠️  Error predicting '{query}': {e}")
                return None

        queries = ["q1", "q2", "q3"]
        tasks = [safe_predict(q) for q in queries]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_results = [r for r in results if r is not None]
        return valid_results

    results = asyncio.run(error_resilient_predictions())
    print(f"   ✅ Completed with {len(results)} valid results")
    print("   📝 Use for: Batch operations that shouldn't fail on single errors")

    # Pattern 3: Background task processing
    print("\n3️⃣  Pattern 3: Background task queue")

    async def background_queue_pattern():
        queue = asyncio.Queue()

        # Producer
        async def producer():
            for i in range(10):
                await queue.put(f"Query {i}")
                await asyncio.sleep(0.01)
            await queue.put(None)  # Sentinel

        # Consumer
        async def consumer():
            results = []
            while True:
                query = await queue.get()
                if query is None:
                    break

                result = await predictor.predict_async(query)
                results.append(result)

            return results

        # Run producer and consumer concurrently
        producer_task = asyncio.create_task(producer())
        consumer_task = asyncio.create_task(consumer())

        await producer_task
        results = await consumer_task

        return results

    results = asyncio.run(background_queue_pattern())
    print(f"   ✅ Processed {len(results)} queries from queue")
    print("   📝 Use for: Background job processing, task queues")

    print("\n💡 Recommendations:")
    print("   - Use cache for high-traffic async APIs")
    print("   - Add timeouts to prevent hanging requests")
    print("   - Use semaphores for rate limiting")
    print("   - Handle errors gracefully with try/except or return_exceptions")
    print("   - Consider asyncio.Queue for background processing")


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
            example_1_basic_async()
        elif example_num == "2":
            example_2_concurrent_predictions()
        elif example_num == "3":
            example_3_async_batch()
        elif example_num == "4":
            example_4_async_with_cache()
        elif example_num == "5":
            example_5_async_web_integration()
        elif example_num == "6":
            example_6_async_streaming()
        elif example_num == "7":
            example_7_async_with_timeout()
        elif example_num == "8":
            example_8_async_event_loop_integration()
        elif example_num == "9":
            example_9_production_async_patterns()
        else:
            print("Usage: python async_prediction_example.py [1-9]")
    else:
        # Run all examples
        example_1_basic_async()
        example_2_concurrent_predictions()
        example_3_async_batch()
        example_4_async_with_cache()
        example_5_async_web_integration()
        example_6_async_streaming()
        example_7_async_with_timeout()
        example_8_async_event_loop_integration()
        example_9_production_async_patterns()

        print("\n" + "=" * 80)
        print("✅ All async prediction examples completed!")
        print("=" * 80)
