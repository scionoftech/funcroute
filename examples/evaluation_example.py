"""
Evaluation Example

Demonstrates how to evaluate a trained FuncRoute router with comprehensive
metrics and visualizations.
"""

from funcroute import FuncRoute, TrainingConfig
from funcroute.core.config import ToolDefinition
from funcroute.data.generator import SyntheticDataGenerator
from funcroute.data.splitter import PatternGroupSplitter
from funcroute.data.validator import DataValidator
from funcroute.evaluation import Evaluator, Visualizer


def main():
    print("=" * 80)
    print("FuncRoute Evaluation Example")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Generate and prepare data
    # -------------------------------------------------------------------------
    print("\nStep 1: Generating synthetic data...")

    tools = [
        ToolDefinition(
            name="manage_order",
            signature="manage_order(order_id: str) -> dict",
            description="Track and manage customer orders",
            examples=[
                "Where is my order?",
                "Track my package",
                "Cancel order",
                "Update my order",
            ],
            keywords=["order", "track", "package", "cancel"],
        ),
        ToolDefinition(
            name="search_products",
            signature="search_products(query: str) -> list",
            description="Search for products",
            examples=[
                "Show me red dresses",
                "Find laptops",
                "Do you have iPhone 15?",
                "Looking for shoes",
            ],
            keywords=["search", "find", "show", "looking for"],
        ),
        ToolDefinition(
            name="process_return",
            signature="process_return(order_id: str) -> dict",
            description="Process returns and refunds",
            examples=[
                "I want to return this",
                "Get a refund",
                "Wrong item received",
                "Item is damaged",
            ],
            keywords=["return", "refund", "wrong", "damaged"],
        ),
        ToolDefinition(
            name="get_product_info",
            signature="get_product_info(product_id: str) -> dict",
            description="Get detailed product information",
            examples=[
                "Tell me about this product",
                "What are the specs?",
                "Product details please",
            ],
            keywords=["product", "details", "specs", "information"],
        ),
    ]

    # Generate data
    generator = SyntheticDataGenerator(method="rule_based")
    data = generator.generate(tools, num_variations=50, num_samples=1000)

    print(f"  Generated {len(data)} samples")

    # Split data
    splitter = PatternGroupSplitter()
    train_data, val_data, test_data = splitter.split(
        data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # -------------------------------------------------------------------------
    # Step 2: Train router (or load existing one)
    # -------------------------------------------------------------------------
    print("\nStep 2: Training router...")

    router = FuncRoute()

    # For this example, we'll train a minimal router
    # In practice, you'd use more epochs and better config
    router.train(
        train_data=train_data,
        val_data=val_data,
        tools=tools,
        config=TrainingConfig(
            output_dir="./evaluation_router",
            num_epochs=1,  # Use more for better results
            batch_size=4,
            save_steps=100,
        ),
    )

    # -------------------------------------------------------------------------
    # Step 3: Evaluate on test data
    # -------------------------------------------------------------------------
    print("\nStep 3: Evaluating router on test data...")

    evaluator = Evaluator(router)
    results = evaluator.evaluate(test_data, verbose=True)

    # Print comprehensive report
    evaluator.print_report(results)

    # -------------------------------------------------------------------------
    # Step 4: Create visualizations
    # -------------------------------------------------------------------------
    print("\nStep 4: Creating visualizations...")

    viz = Visualizer()

    if viz.available:
        # Confusion matrix
        print("  Creating confusion matrix...")
        viz.plot_confusion_matrix(
            results, save_path="./confusion_matrix.png", normalize=False
        )

        # Normalized confusion matrix
        print("  Creating normalized confusion matrix...")
        viz.plot_confusion_matrix(
            results, save_path="./confusion_matrix_normalized.png", normalize=True
        )

        # Confidence distribution
        print("  Creating confidence distribution...")
        viz.plot_confidence_distribution(results, save_path="./confidence_dist.png")

        # Calibration curve
        print("  Creating calibration curve...")
        viz.plot_calibration_curve(results, save_path="./calibration.png")

        # Per-tool performance
        print("  Creating per-tool performance chart...")
        viz.plot_per_tool_performance(
            results, metric="f1", save_path="./per_tool_performance.png"
        )

        # Latency distribution
        print("  Creating latency distribution...")
        viz.plot_latency_distribution(results, save_path="./latency_dist.png")

        # Comprehensive dashboard
        print("  Creating evaluation dashboard...")
        viz.create_evaluation_dashboard(results, save_path="./eval_dashboard.png")

        print("\n✅ All visualizations saved!")
    else:
        print("\n⚠️  Visualizations not available (install matplotlib and seaborn)")

    # -------------------------------------------------------------------------
    # Step 5: Analyze specific metrics
    # -------------------------------------------------------------------------
    print("\nStep 5: Detailed metric analysis...")

    # Accuracy breakdown
    print(f"\n📊 Accuracy Metrics:")
    print(f"  Overall Accuracy:    {results['accuracy']:.2%}")
    print(f"  Top-3 Accuracy:      {results['top3_accuracy']:.2%}")
    print(f"  Top-5 Accuracy:      {results['top5_accuracy']:.2%}")

    # Best and worst performing tools
    tool_metrics = results["per_tool_metrics"]
    sorted_tools = sorted(tool_metrics.items(), key=lambda x: x[1]["f1"], reverse=True)

    print(f"\n🏆 Best Performing Tools (by F1):")
    for tool, metrics in sorted_tools[:3]:
        print(f"  {tool:30s} F1: {metrics['f1']:.2%} (n={metrics['support']})")

    print(f"\n📉 Worst Performing Tools (by F1):")
    for tool, metrics in sorted_tools[-3:]:
        print(f"  {tool:30s} F1: {metrics['f1']:.2%} (n={metrics['support']})")

    # Calibration analysis
    print(f"\n📐 Calibration Analysis:")
    print(f"  ECE (Expected Calibration Error): {results['calibration']['ece']:.4f}")
    if results["calibration"]["ece"] < 0.05:
        print(f"  ✅ Excellent calibration!")
    elif results["calibration"]["ece"] < 0.10:
        print(f"  ✅ Good calibration")
    elif results["calibration"]["ece"] < 0.15:
        print(f"  ⚠️  Fair calibration - could be improved")
    else:
        print(f"  ❌ Poor calibration - needs improvement")

    # Latency analysis
    print(f"\n⚡ Latency Analysis:")
    lat = results["latency"]
    print(f"  Mean latency:   {lat['mean']:.2f}ms")
    print(f"  P95 latency:    {lat['p95']:.2f}ms")
    print(f"  P99 latency:    {lat['p99']:.2f}ms")

    if lat["p95"] < 50:
        print(f"  ✅ Excellent latency!")
    elif lat["p95"] < 100:
        print(f"  ✅ Good latency")
    elif lat["p95"] < 200:
        print(f"  ⚠️  Fair latency - could be optimized")
    else:
        print(f"  ❌ High latency - needs optimization")

    # -------------------------------------------------------------------------
    # Step 6: Error analysis
    # -------------------------------------------------------------------------
    print("\nStep 6: Error analysis...")

    predictions = results["predictions"]
    ground_truth = results["ground_truth"]
    confidences = results["confidences"]

    # Find misclassified samples
    errors = []
    for i, (pred, true, conf) in enumerate(
        zip(predictions, ground_truth, confidences)
    ):
        if pred != true:
            errors.append(
                {
                    "query": test_data[i]["query"],
                    "predicted": pred,
                    "actual": true,
                    "confidence": conf,
                }
            )

    print(f"\n❌ Misclassified Samples: {len(errors)} / {len(test_data)}")

    if errors:
        print(f"\nShowing first 5 errors:")
        for i, error in enumerate(errors[:5], 1):
            print(f"\n  {i}. Query: '{error['query']}'")
            print(f"     Predicted: {error['predicted']} (conf: {error['confidence']:.2%})")
            print(f"     Actual: {error['actual']}")

    # High-confidence errors (concerning)
    high_conf_errors = [e for e in errors if e["confidence"] > 0.8]
    if high_conf_errors:
        print(f"\n⚠️  {len(high_conf_errors)} high-confidence errors (confidence > 80%):")
        for error in high_conf_errors[:3]:
            print(
                f"     '{error['query']}': {error['predicted']} (should be {error['actual']})"
            )

    # -------------------------------------------------------------------------
    # Step 7: Recommendations
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if results["accuracy"] < 0.85:
        print("\n📝 Model Performance:")
        print("  - Consider training for more epochs")
        print("  - Add more diverse training examples")
        print("  - Review misclassified samples for patterns")

    if results["calibration"]["ece"] > 0.10:
        print("\n📝 Calibration:")
        print("  - Model confidence scores are not well-calibrated")
        print("  - Consider temperature scaling or calibration techniques")

    if lat["p95"] > 100:
        print("\n📝 Latency:")
        print("  - Consider model optimization techniques")
        print("  - Use quantization or pruning")
        print("  - Implement caching for common queries")

    # Check for class imbalance issues
    supports = [m["support"] for m in tool_metrics.values()]
    if max(supports) / min(supports) > 3:
        print("\n📝 Data Balance:")
        print("  - Training data has class imbalance")
        print("  - Consider balancing the dataset")
        print("  - Use weighted loss during training")

    print("\n" + "=" * 80)


def cross_validation_example():
    """
    Example of cross-validation.

    NOTE: This requires training multiple models and is time-consuming.
    """
    print("\n" + "=" * 80)
    print("Cross-Validation Example")
    print("=" * 80)

    # Generate data
    tools = [
        ToolDefinition(
            name="tool1",
            signature="tool1() -> str",
            description="Tool 1",
            examples=["query 1", "test 1"],
        ),
        ToolDefinition(
            name="tool2",
            signature="tool2() -> str",
            description="Tool 2",
            examples=["query 2", "test 2"],
        ),
    ]

    generator = SyntheticDataGenerator(method="rule_based")
    data = generator.generate(tools, num_variations=20, num_samples=200)

    # Define training function
    def train_func(train_data):
        router = FuncRoute()
        router.train(
            train_data,
            tools=tools,  # CRITICAL: Must provide tools!
            config=TrainingConfig(
                output_dir="./cv_router_temp",
                num_epochs=1,
                batch_size=4,
            ),
        )
        return router

    # Run cross-validation
    evaluator = Evaluator()
    cv_results = evaluator.cross_validate(data, train_func, n_folds=3, verbose=True)

    print(f"\n✅ Cross-Validation Complete!")
    print(f"   Mean Accuracy: {cv_results['accuracy_mean']:.2%} ± {cv_results['accuracy_std']:.2%}")


if __name__ == "__main__":
    # Run main evaluation example
    main()

    # Uncomment to run cross-validation example (time-consuming)
    # cross_validation_example()
