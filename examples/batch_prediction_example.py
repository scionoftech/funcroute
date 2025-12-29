"""
Batch Prediction Example

Demonstrates efficient batch prediction with parallelization and progress tracking.
"""

from funcroute import FuncRoute, TrainingConfig
from funcroute.core.config import ToolDefinition
from funcroute.data.generator import SyntheticDataGenerator
from funcroute.data.splitter import PatternGroupSplitter
from funcroute.inference import Predictor, RouteCache
import time


def prepare_model():
    """Prepare a simple model for examples"""
    print("Setting up example model...")

    tools = [
        ToolDefinition(
            name="manage_order",
            signature="manage_order(order_id: str) -> dict",
            description="Track and manage orders",
            examples=["Where is my order?", "Track package"],
        ),
        ToolDefinition(
            name="search_products",
            signature="search_products(query: str) -> list",
            description="Search products",
            examples=["Find laptops", "Show shoes"],
        ),
        ToolDefinition(
            name="process_return",
            signature="process_return(order_id: str) -> dict",
            description="Process returns",
            examples=["Return this", "Get refund"],
        ),
    ]

    # Generate and train
    generator = SyntheticDataGenerator(method="rule_based")
    data = generator.generate(tools, num_variations=20, num_samples=200)
    splitter = PatternGroupSplitter()
    train_data, val_data, _ = splitter.split(data)

    router = FuncRoute()
    router.train(
        train_data=train_data,
        val_data=val_data,
        tools=tools,
        config=TrainingConfig(
            output_dir="./batch_example_model",
            num_epochs=1,
            batch_size=4,
        ),
    )

    print("✅ Model ready!\n")
    return router


def example_1_basic_batch():
    """
    Example 1: Basic batch prediction

    Process multiple queries in parallel.
    """
    print("=" * 80)
    print("Example 1: Basic Batch Prediction")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    # Prepare queries
    queries = [
        "Where is my order #12345?",
        "Find wireless headphones under $100",
        "I want to return my laptop",
        "Track my package",
        "Show me running shoes",
        "Cancel my order",
        "What's the status of order #67890?",
        "Search for iPhone cases",
        "Return defective item",
        "Where is my shipment?",
    ]

    print(f"\nPredicting {len(queries)} queries in batch...\n")

    # Batch predict with progress bar
    start = time.time()
    results = predictor.predict_batch(queries, show_progress=True)
    elapsed = time.time() - start

    # Display results
    print(f"\n{'Query':<40s} {'Tool':<20s} {'Confidence':>12s} {'Latency':>10s}")
    print("-" * 85)

    for result in results:
        print(
            f"{result.query[:37]+'...' if len(result.query) > 40 else result.query:<40s} "
            f"{result.tool:<20s} "
            f"{result.confidence:>11.1%} "
            f"{result.latency_ms:>9.1f}ms"
        )

    print(f"\n✅ Batch completed in {elapsed:.2f}s")
    print(f"   Average latency: {sum(r.latency_ms for r in results) / len(results):.1f}ms per query")
    print(f"   Throughput: {len(queries) / elapsed:.1f} queries/second")


def example_2_parallel_workers():
    """
    Example 2: Controlling parallelization

    Adjust max_workers for different performance characteristics.
    """
    print("\n" + "=" * 80)
    print("Example 2: Controlling Parallelization")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    # Large batch
    queries = [f"Query {i}: Find product type {i % 10}" for i in range(100)]

    # Test different worker counts
    worker_counts = [1, 2, 4, 8]

    print(f"\nProcessing {len(queries)} queries with different worker counts:\n")

    for workers in worker_counts:
        start = time.time()
        results = predictor.predict_batch(
            queries, max_workers=workers, show_progress=False
        )
        elapsed = time.time() - start

        throughput = len(queries) / elapsed
        print(f"  Workers: {workers:2d}  |  Time: {elapsed:6.2f}s  |  Throughput: {throughput:6.1f} queries/s")

    print("\n💡 Tip: More workers = faster for I/O-bound tasks, but diminishing returns")


def example_3_batch_with_filtering():
    """
    Example 3: Batch prediction with filtering

    Filter results based on confidence threshold.
    """
    print("\n" + "=" * 80)
    print("Example 3: Batch Prediction with Filtering")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    queries = [
        "Where is my order?",
        "Find laptops",
        "Return this item",
        "xyz123",  # Low confidence query
        "Track package",
        "asdfghjkl",  # Low confidence query
    ]

    print(f"\nFiltering results with confidence > 80%...\n")

    # Filter high-confidence results
    high_confidence = predictor.predict_with_filter(
        queries, filter_fn=lambda r: r.confidence > 0.8
    )

    print(f"Results with high confidence ({len(high_confidence)}/{len(queries)}):")
    for result in high_confidence:
        print(f"  ✅ '{result.query}' → {result.tool} ({result.confidence:.1%})")

    # Show filtered out
    all_results = predictor.predict_batch(queries, show_progress=False)
    low_confidence = [r for r in all_results if r.confidence <= 0.8]

    if low_confidence:
        print(f"\nFiltered out ({len(low_confidence)}/{len(queries)}):")
        for result in low_confidence:
            print(f"  ⚠️  '{result.query}' → {result.tool} ({result.confidence:.1%})")


def example_4_batch_with_callback():
    """
    Example 4: Real-time processing with callbacks

    Process results as they complete using callbacks.
    """
    print("\n" + "=" * 80)
    print("Example 4: Batch Prediction with Callbacks")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    queries = [
        "Where is my order?",
        "Find wireless keyboards",
        "Return defective product",
        "Track shipment #123",
        "Search for monitors",
    ]

    # Statistics collector
    stats = {"total": 0, "by_tool": {}}

    def process_result(result):
        """Callback to process each result"""
        stats["total"] += 1

        # Count by tool
        if result.tool not in stats["by_tool"]:
            stats["by_tool"][result.tool] = 0
        stats["by_tool"][result.tool] += 1

        # Print as we go
        print(f"  [{stats['total']}/{len(queries)}] '{result.query}' → {result.tool}")

    print(f"\nProcessing {len(queries)} queries with callback...\n")

    predictor.predict_with_callback(queries, callback=process_result)

    print(f"\n📊 Statistics:")
    print(f"   Total processed: {stats['total']}")
    print(f"   By tool:")
    for tool, count in sorted(stats["by_tool"].items()):
        print(f"     - {tool}: {count}")


def example_5_batch_error_handling():
    """
    Example 5: Error handling in batch prediction

    Gracefully handle errors without stopping entire batch.
    """
    print("\n" + "=" * 80)
    print("Example 5: Error Handling in Batch Prediction")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    # Mix of valid and potentially problematic queries
    queries = [
        "Where is my order?",
        "",  # Empty query
        "Find laptops",
        " ",  # Whitespace only
        "Return this item",
        "a" * 10000,  # Very long query
    ]

    print(f"\nProcessing batch with potential error cases...\n")

    results = predictor.predict_batch(queries, show_progress=False)

    # Check results
    successful = 0
    errors = 0

    for i, result in enumerate(results):
        if result.tool == "error" or result.confidence == 0.0:
            errors += 1
            error_msg = result.metadata.get("error", "Unknown") if result.metadata else "Unknown"
            print(f"  ❌ Query {i+1}: Error - {error_msg}")
        else:
            successful += 1
            print(f"  ✅ Query {i+1}: '{result.query[:30]}...' → {result.tool}")

    print(f"\n📊 Results: {successful} successful, {errors} errors")
    print("💡 Tip: Errors don't stop the entire batch!")


def example_6_performance_comparison():
    """
    Example 6: Sequential vs Batch performance comparison

    Demonstrate the speed improvement of batch processing.
    """
    print("\n" + "=" * 80)
    print("Example 6: Sequential vs Batch Performance")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    # Test queries
    queries = [f"Find product {i}" for i in range(50)]

    # Sequential processing
    print(f"\nProcessing {len(queries)} queries sequentially...")
    start = time.time()
    sequential_results = []
    for query in queries:
        result = predictor._predict_single(query, use_cache=False)
        sequential_results.append(result)
    sequential_time = time.time() - start

    # Batch processing
    print(f"Processing {len(queries)} queries in batch...")
    start = time.time()
    batch_results = predictor.predict_batch(queries, show_progress=False, use_cache=False)
    batch_time = time.time() - start

    # Comparison
    speedup = sequential_time / batch_time

    print(f"\n📊 Performance Comparison:")
    print(f"   Sequential: {sequential_time:.2f}s ({len(queries)/sequential_time:.1f} queries/s)")
    print(f"   Batch:      {batch_time:.2f}s ({len(queries)/batch_time:.1f} queries/s)")
    print(f"   Speedup:    {speedup:.2f}x faster! 🚀")


def example_7_batch_with_progress():
    """
    Example 7: Custom progress tracking

    Monitor progress with detailed information.
    """
    print("\n" + "=" * 80)
    print("Example 7: Detailed Progress Tracking")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    queries = [f"Query {i}: Search for item {i}" for i in range(20)]

    print(f"\nProcessing {len(queries)} queries with progress bar...\n")

    # Batch predict with progress
    results = predictor.predict_batch(queries, show_progress=True)

    # Analyze results
    avg_confidence = sum(r.confidence for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)

    print(f"\n📊 Summary:")
    print(f"   Queries processed: {len(results)}")
    print(f"   Average confidence: {avg_confidence:.1%}")
    print(f"   Average latency: {avg_latency:.1f}ms")

    # Tool distribution
    from collections import Counter
    tool_counts = Counter(r.tool for r in results)
    print(f"\n   Tool distribution:")
    for tool, count in tool_counts.most_common():
        print(f"     - {tool}: {count} ({count/len(results):.1%})")


if __name__ == "__main__":
    import sys

    # Prepare model (only once)
    import os
    if not os.path.exists("./batch_example_model"):
        prepare_model()
    else:
        print("Using existing model...\n")

    # Run examples
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_1_basic_batch()
        elif example_num == "2":
            example_2_parallel_workers()
        elif example_num == "3":
            example_3_batch_with_filtering()
        elif example_num == "4":
            example_4_batch_with_callback()
        elif example_num == "5":
            example_5_batch_error_handling()
        elif example_num == "6":
            example_6_performance_comparison()
        elif example_num == "7":
            example_7_batch_with_progress()
        else:
            print("Usage: python batch_prediction_example.py [1-7]")
    else:
        # Run all examples
        example_1_basic_batch()
        example_2_parallel_workers()
        example_3_batch_with_filtering()
        example_4_batch_with_callback()
        example_5_batch_error_handling()
        example_6_performance_comparison()
        example_7_batch_with_progress()

        print("\n" + "=" * 80)
        print("✅ All batch prediction examples completed!")
        print("=" * 80)
