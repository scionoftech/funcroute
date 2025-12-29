"""
Command-line interface for FuncRoute
"""

import argparse
import sys
import json
from pathlib import Path


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="FuncRoute - Intelligent Function/Tool Routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a routing model")
    train_parser.add_argument(
        "--data", required=True, help="Path to training data (JSONL format)"
    )
    train_parser.add_argument("--output", required=True, help="Output directory for model")
    train_parser.add_argument("--val-data", help="Path to validation data (optional)")
    train_parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    train_parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    train_parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    train_parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model", required=True, help="Path to trained model")
    eval_parser.add_argument("--data", required=True, help="Path to test data (JSONL)")
    eval_parser.add_argument(
        "--output", help="Output directory for evaluation results"
    )
    eval_parser.add_argument(
        "--visualize", action="store_true", help="Create visualizations"
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict routes for queries")
    predict_parser.add_argument("--model", required=True, help="Path to trained model")
    predict_parser.add_argument("--query", help="Single query to route")
    predict_parser.add_argument("--queries-file", help="File with queries (one per line)")
    predict_parser.add_argument(
        "--output", help="Output file for predictions (JSON)"
    )
    predict_parser.add_argument(
        "--cache-size", type=int, default=1000, help="Cache size"
    )

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start REST API server")
    serve_parser.add_argument("--model", required=True, help="Path to trained model")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve_parser.add_argument(
        "--cache-size", type=int, default=1000, help="Cache size (0 = disabled)"
    )
    serve_parser.add_argument(
        "--cache-ttl", type=int, default=300, help="Cache TTL in seconds"
    )
    serve_parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching"
    )

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic training data")
    gen_parser.add_argument("--tools", required=True, help="Path to tool definitions (JSON)")
    gen_parser.add_argument("--output", required=True, help="Output file for generated data (JSONL)")
    gen_parser.add_argument(
        "--num-samples", type=int, default=1000, help="Number of samples to generate"
    )
    gen_parser.add_argument(
        "--num-variations", type=int, default=50, help="Variations per pattern"
    )
    gen_parser.add_argument(
        "--method", choices=["rule_based", "llm_based"], default="rule_based",
        help="Generation method"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "serve":
        serve_command(args)
    elif args.command == "generate":
        generate_command(args)


def train_command(args):
    """Execute train command"""
    from funcroute import FuncRoute, TrainingConfig
    from funcroute.data.loader import DataLoader

    print(f"🚀 Training FuncRoute model")
    print(f"   Data: {args.data}")
    print(f"   Output: {args.output}")

    # Load data
    loader = DataLoader()
    train_data = loader.load(args.data)
    print(f"   Loaded {len(train_data)} training samples")

    val_data = None
    if args.val_data:
        val_data = loader.load(args.val_data)
        print(f"   Loaded {len(val_data)} validation samples")

    # Create config
    config = TrainingConfig(
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    # Train
    router = FuncRoute()
    router.train(train_data, val_data=val_data, config=config)

    print(f"\n✅ Training complete! Model saved to {args.output}")


def evaluate_command(args):
    """Execute evaluate command"""
    from funcroute import FuncRoute
    from funcroute.data.loader import DataLoader
    from funcroute.evaluation import Evaluator, Visualizer

    print(f"📊 Evaluating FuncRoute model")
    print(f"   Model: {args.model}")
    print(f"   Data: {args.data}")

    # Load model
    router = FuncRoute.load(args.model)
    print(f"   ✅ Model loaded")

    # Load test data
    loader = DataLoader()
    test_data = loader.load(args.data)
    print(f"   Loaded {len(test_data)} test samples")

    # Evaluate
    evaluator = Evaluator(router)
    results = evaluator.evaluate(test_data, verbose=True)

    # Print report
    evaluator.print_report(results)

    # Save results
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            # Convert numpy types to native Python types
            import numpy as np
            metrics = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results.items()
                if k not in ["predictions", "ground_truth", "confidences"]
            }
            json.dump(metrics, f, indent=2, default=str)
        print(f"\n   Saved metrics to {metrics_file}")

        # Create visualizations
        if args.visualize:
            viz = Visualizer()
            if viz.available:
                print(f"\n   Creating visualizations...")
                viz.plot_confusion_matrix(
                    results, save_path=str(output_dir / "confusion_matrix.png")
                )
                viz.plot_calibration_curve(
                    results, save_path=str(output_dir / "calibration.png")
                )
                viz.create_evaluation_dashboard(
                    results, save_path=str(output_dir / "dashboard.png")
                )
                print(f"   ✅ Visualizations saved to {output_dir}")
            else:
                print(f"   ⚠️  Visualizations unavailable (install matplotlib and seaborn)")

    print(f"\n✅ Evaluation complete!")


def predict_command(args):
    """Execute predict command"""
    from funcroute import FuncRoute
    from funcroute.inference import Predictor, RouteCache

    print(f"🔮 Making predictions")
    print(f"   Model: {args.model}")

    # Load model
    router = FuncRoute.load(args.model)

    # Create predictor with cache
    cache = RouteCache(max_size=args.cache_size) if args.cache_size > 0 else None
    predictor = Predictor(router, cache=cache)

    # Get queries
    queries = []
    if args.query:
        queries = [args.query]
    elif args.queries_file:
        with open(args.queries_file) as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Provide either --query or --queries-file")
        sys.exit(1)

    print(f"   Predicting {len(queries)} queries...")

    # Predict
    results = predictor.predict_batch(queries, show_progress=True)

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            output_data = [
                {
                    "query": r.query,
                    "tool": r.tool,
                    "confidence": r.confidence,
                    "latency_ms": r.latency_ms,
                }
                for r in results
            ]
            json.dump(output_data, f, indent=2)
        print(f"   ✅ Predictions saved to {args.output}")
    else:
        # Print to stdout
        for result in results:
            print(f"\n   Query: {result.query}")
            print(f"   Tool: {result.tool} (confidence: {result.confidence:.2%})")
            print(f"   Latency: {result.latency_ms:.1f}ms")

    # Cache stats
    if cache:
        stats = predictor.get_cache_stats()
        print(f"\n   Cache stats: {stats['hit_rate']:.1%} hit rate, {stats['size']} entries")


def serve_command(args):
    """Execute serve command"""
    from funcroute import FuncRoute
    from funcroute.inference.server import run_server

    print(f"🌐 Starting FuncRoute server")
    print(f"   Model: {args.model}")

    # Load model
    router = FuncRoute.load(args.model)
    print(f"   ✅ Model loaded")

    # Determine cache settings
    cache_size = None if args.no_cache else args.cache_size
    cache_ttl = args.cache_ttl if not args.no_cache else None

    # Run server
    run_server(
        router,
        host=args.host,
        port=args.port,
        cache_size=cache_size,
        cache_ttl=cache_ttl,
    )


def generate_command(args):
    """Execute generate command"""
    from funcroute.core.config import ToolDefinition
    from funcroute.data.generator import SyntheticDataGenerator
    from funcroute.data.loader import DataLoader

    print(f"🎲 Generating synthetic training data")
    print(f"   Tools: {args.tools}")
    print(f"   Output: {args.output}")

    # Load tool definitions
    with open(args.tools) as f:
        tools_data = json.load(f)

    tools = [ToolDefinition(**tool) for tool in tools_data]
    print(f"   Loaded {len(tools)} tool definitions")

    # Generate
    generator = SyntheticDataGenerator(method=args.method)
    data = generator.generate(
        tools,
        num_variations=args.num_variations,
        num_samples=args.num_samples,
    )

    print(f"   Generated {len(data)} samples")

    # Save
    loader = DataLoader()
    # Note: DataLoader doesn't have a save method yet, so use manual JSONL write
    with open(args.output, "w") as f:
        for sample in data:
            f.write(json.dumps(sample) + "\n")

    print(f"   ✅ Saved to {args.output}")


if __name__ == "__main__":
    main()
