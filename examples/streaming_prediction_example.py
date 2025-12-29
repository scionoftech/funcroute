"""
Streaming Prediction Example

Demonstrates streaming predictions for processing results as they become available.
Useful for large batches where you want to start processing before all predictions complete.
"""

from funcroute import FuncRoute
from funcroute.inference import Predictor, RouteCache
import time
from collections import defaultdict


def example_1_basic_streaming():
    """
    Example 1: Basic streaming prediction

    Process results as they complete (order not guaranteed).
    """
    print("=" * 80)
    print("Example 1: Basic Streaming Prediction")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    queries = [
        "Where is my order?",
        "Find wireless keyboards",
        "Return defective product",
        "Track my package",
        "Search for monitors",
        "Cancel my order",
        "Product details for item #123",
        "Where is my shipment?",
    ]

    print(f"\nStreaming {len(queries)} predictions...\n")
    print(f"{'Completed':<12s} {'Query':<35s} {'Tool':<20s} {'Conf':>8s}")
    print("-" * 80)

    start = time.time()
    completed = 0

    # Stream results as they complete
    for result in predictor.predict_stream(queries):
        completed += 1
        elapsed = time.time() - start

        print(
            f"{completed}/{len(queries):<7s} "
            f"{result.query[:32]+'...' if len(result.query) > 35 else result.query:<35s} "
            f"{result.tool:<20s} "
            f"{result.confidence:>7.1%}"
        )

    total_time = time.time() - start
    print(f"\n✅ All {len(queries)} predictions completed in {total_time:.2f}s")
    print("💡 Notice: Results may arrive out of order!")


def example_2_streaming_with_early_processing():
    """
    Example 2: Early processing with streaming

    Start processing results before all predictions complete.
    """
    print("\n" + "=" * 80)
    print("Example 2: Early Processing with Streaming")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    # Simulate large batch
    queries = [f"Find product type {i % 5}" for i in range(50)]

    # Statistics accumulator
    stats = {
        "total": 0,
        "by_tool": defaultdict(int),
        "high_confidence": 0,
        "start_time": time.time(),
    }

    print(f"\nStreaming {len(queries)} predictions with real-time processing...\n")

    # Process as results arrive
    for result in predictor.predict_stream(queries):
        stats["total"] += 1
        stats["by_tool"][result.tool] += 1

        if result.confidence > 0.9:
            stats["high_confidence"] += 1

        # Print every 10 results
        if stats["total"] % 10 == 0:
            elapsed = time.time() - stats["start_time"]
            rate = stats["total"] / elapsed
            print(f"  [{stats['total']:2d}/{len(queries)}] Processed at {rate:.1f} queries/s")

    total_time = time.time() - stats["start_time"]

    print(f"\n📊 Final Statistics:")
    print(f"   Total: {stats['total']}")
    print(f"   High confidence: {stats['high_confidence']} ({stats['high_confidence']/stats['total']:.1%})")
    print(f"   Time: {total_time:.2f}s")
    print(f"\n   By tool:")
    for tool, count in sorted(stats["by_tool"].items(), key=lambda x: x[1], reverse=True):
        print(f"     - {tool}: {count}")


def example_3_streaming_vs_batch():
    """
    Example 3: Streaming vs Batch comparison

    Show when streaming is beneficial.
    """
    print("\n" + "=" * 80)
    print("Example 3: Streaming vs Batch Comparison")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    queries = [f"Query {i}" for i in range(30)]

    # Batch mode - must wait for all
    print("\n1️⃣  Batch mode (wait for all results):")
    start = time.time()
    batch_results = predictor.predict_batch(queries, show_progress=False)
    batch_time = time.time() - start
    print(f"   Time to first result: {batch_time:.2f}s")
    print(f"   Time to all results:  {batch_time:.2f}s")

    # Streaming mode - process as available
    print("\n2️⃣  Streaming mode (process as available):")
    start = time.time()
    first_result_time = None
    stream_count = 0

    for result in predictor.predict_stream(queries):
        if first_result_time is None:
            first_result_time = time.time() - start
        stream_count += 1

    stream_time = time.time() - start

    print(f"   Time to first result: {first_result_time:.2f}s")
    print(f"   Time to all results:  {stream_time:.2f}s")

    print(f"\n💡 Key Insight:")
    print(f"   Streaming gives you first result ~{batch_time/first_result_time:.1f}x faster!")
    print(f"   Perfect for real-time processing or large batches")


def example_4_streaming_with_timeout():
    """
    Example 4: Streaming with early termination

    Stop processing after collecting enough results.
    """
    print("\n" + "=" * 80)
    print("Example 4: Streaming with Early Termination")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    # Large query set
    queries = [f"Search for item {i}" for i in range(100)]

    print(f"\nSearching {len(queries)} queries for high-confidence 'search_products' results...")
    print("Stopping after finding 5 matches.\n")

    # Use predict_until to stop when condition met
    results = predictor.predict_until(
        queries,
        condition=lambda results: len([
            r for r in results
            if r.tool == "search_products" and r.confidence > 0.85
        ]) >= 5
    )

    # Filter matches
    matches = [r for r in results if r.tool == "search_products" and r.confidence > 0.85]

    print(f"✅ Found {len(matches)} matching results after checking {len(results)}/{len(queries)} queries:")
    for i, result in enumerate(matches, 1):
        print(f"   {i}. '{result.query}' (confidence: {result.confidence:.1%})")

    print(f"\n💡 Saved {len(queries) - len(results)} unnecessary predictions!")


def example_5_streaming_pipeline():
    """
    Example 5: Multi-stage streaming pipeline

    Build a processing pipeline with streaming.
    """
    print("\n" + "=" * 80)
    print("Example 5: Multi-Stage Streaming Pipeline")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    queries = [
        "Where is my order #123?",
        "Find gaming laptops",
        "Return broken headphones",
        "Track package #456",
        "Search for monitors",
        "Cancel order #789",
    ]

    # Pipeline stages
    class Pipeline:
        def __init__(self):
            self.stage1_results = []
            self.stage2_results = []
            self.stage3_results = []

        def stage1_validate(self, result):
            """Stage 1: Validate confidence"""
            self.stage1_results.append(result)
            if result.confidence > 0.7:
                print(f"  ✅ Stage 1: '{result.query}' passed validation")
                return True
            else:
                print(f"  ❌ Stage 1: '{result.query}' failed validation (low confidence)")
                return False

        def stage2_categorize(self, result):
            """Stage 2: Categorize by tool"""
            self.stage2_results.append(result)
            print(f"     📁 Stage 2: Categorized as '{result.tool}'")
            return True

        def stage3_process(self, result):
            """Stage 3: Final processing"""
            self.stage3_results.append(result)
            print(f"        🔄 Stage 3: Processed successfully")

    pipeline = Pipeline()

    print(f"\nProcessing {len(queries)} queries through 3-stage pipeline:\n")

    # Stream and process through pipeline
    for result in predictor.predict_stream(queries):
        if pipeline.stage1_validate(result):
            if pipeline.stage2_categorize(result):
                pipeline.stage3_process(result)

    print(f"\n📊 Pipeline Statistics:")
    print(f"   Stage 1 (Validation):   {len(pipeline.stage1_results)} processed")
    print(f"   Stage 2 (Categorize):   {len(pipeline.stage2_results)} processed")
    print(f"   Stage 3 (Final):        {len(pipeline.stage3_results)} processed")


def example_6_streaming_aggregation():
    """
    Example 6: Real-time aggregation with streaming

    Compute statistics in real-time as results arrive.
    """
    print("\n" + "=" * 80)
    print("Example 6: Real-Time Aggregation")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    queries = [f"Query {i}: Find item type {i % 3}" for i in range(40)]

    # Real-time aggregator
    class RealtimeStats:
        def __init__(self):
            self.count = 0
            self.confidence_sum = 0
            self.latency_sum = 0
            self.tools = defaultdict(int)

        def update(self, result):
            self.count += 1
            self.confidence_sum += result.confidence
            self.latency_sum += result.latency_ms
            self.tools[result.tool] += 1

        @property
        def avg_confidence(self):
            return self.confidence_sum / self.count if self.count > 0 else 0

        @property
        def avg_latency(self):
            return self.latency_sum / self.count if self.count > 0 else 0

        def print_stats(self):
            print(
                f"  [{self.count:2d}] "
                f"Avg conf: {self.avg_confidence:.1%}  |  "
                f"Avg latency: {self.avg_latency:.1f}ms  |  "
                f"Tools: {len(self.tools)}"
            )

    stats = RealtimeStats()

    print(f"\nStreaming {len(queries)} predictions with real-time stats:\n")

    for result in predictor.predict_stream(queries):
        stats.update(result)

        # Print stats every 10 results
        if stats.count % 10 == 0:
            stats.print_stats()

    print(f"\n✅ Final Statistics:")
    stats.print_stats()
    print(f"\n   Tool distribution:")
    for tool, count in sorted(stats.tools.items()):
        print(f"     - {tool}: {count} ({count/stats.count:.1%})")


def example_7_streaming_to_database():
    """
    Example 7: Streaming results to storage

    Simulate saving results to database as they arrive.
    """
    print("\n" + "=" * 80)
    print("Example 7: Streaming to Database")
    print("=" * 80)

    router = FuncRoute.load("./batch_example_model")
    predictor = Predictor(router)

    queries = [f"Process transaction {i}" for i in range(20)]

    # Simulated database
    database = []

    def save_to_db(result):
        """Simulate database save"""
        record = {
            "query": result.query,
            "tool": result.tool,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
            "timestamp": time.time(),
        }
        database.append(record)
        # Simulate slow DB write
        time.sleep(0.01)

    print(f"\nStreaming {len(queries)} predictions and saving to database...\n")

    batch_size = 5
    batch_buffer = []

    for result in predictor.predict_stream(queries):
        batch_buffer.append(result)

        # Batch save every 5 results
        if len(batch_buffer) >= batch_size:
            print(f"  💾 Saving batch of {len(batch_buffer)} results to database...")
            for buffered_result in batch_buffer:
                save_to_db(buffered_result)
            print(f"     ✅ Saved! Total in DB: {len(database)}")
            batch_buffer = []

    # Save remaining
    if batch_buffer:
        print(f"  💾 Saving final batch of {len(batch_buffer)} results...")
        for buffered_result in batch_buffer:
            save_to_db(buffered_result)
        print(f"     ✅ Saved! Total in DB: {len(database)}")

    print(f"\n✅ All {len(database)} results saved to database")
    print("💡 Streaming allows incremental saves without waiting for all results")


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
            example_1_basic_streaming()
        elif example_num == "2":
            example_2_streaming_with_early_processing()
        elif example_num == "3":
            example_3_streaming_vs_batch()
        elif example_num == "4":
            example_4_streaming_with_timeout()
        elif example_num == "5":
            example_5_streaming_pipeline()
        elif example_num == "6":
            example_6_streaming_aggregation()
        elif example_num == "7":
            example_7_streaming_to_database()
        else:
            print("Usage: python streaming_prediction_example.py [1-7]")
    else:
        # Run all examples
        example_1_basic_streaming()
        example_2_streaming_with_early_processing()
        example_3_streaming_vs_batch()
        example_4_streaming_with_timeout()
        example_5_streaming_pipeline()
        example_6_streaming_aggregation()
        example_7_streaming_to_database()

        print("\n" + "=" * 80)
        print("✅ All streaming prediction examples completed!")
        print("=" * 80)
