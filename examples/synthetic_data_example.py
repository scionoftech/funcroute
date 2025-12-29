"""
Synthetic Data Generation Example

Demonstrates how to generate synthetic training data using FuncRoute's
data generation utilities.
"""

from funcroute import FuncRoute, TrainingConfig
from funcroute.core.config import ToolDefinition
from funcroute.data.generator import SyntheticDataGenerator
from funcroute.data.validator import DataValidator
from funcroute.data.splitter import PatternGroupSplitter


def main():
    print("=" * 80)
    print("FuncRoute Synthetic Data Generation Example")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Define your tools
    # -------------------------------------------------------------------------
    print("\nStep 1: Defining tools...")

    tools = [
        ToolDefinition(
            name="manage_order",
            signature="manage_order(order_id: str, action: str) -> dict",
            description="Track, update, or cancel customer orders",
            examples=[
                "Where is my order?",
                "Track my package",
                "Cancel order #12345",
                "Update my order",
            ],
            keywords=["order", "track", "package", "cancel", "update", "status"],
        ),
        ToolDefinition(
            name="search_products",
            signature="search_products(query: str, category: str, filters: dict) -> list",
            description="Search for products in the catalog",
            examples=[
                "Show me red dresses",
                "Find laptops under $1000",
                "Do you have iPhone 15?",
                "Looking for running shoes",
            ],
            keywords=["search", "find", "show", "looking for", "do you have"],
        ),
        ToolDefinition(
            name="process_return",
            signature="process_return(order_id: str, reason: str) -> dict",
            description="Process returns and refunds",
            examples=[
                "I want to return this",
                "Get a refund",
                "Wrong item received",
                "Item is damaged",
            ],
            keywords=["return", "refund", "wrong", "damaged", "defective"],
        ),
        ToolDefinition(
            name="get_product_info",
            signature="get_product_info(product_id: str) -> dict",
            description="Get detailed product information and specifications",
            examples=[
                "Tell me about this product",
                "What are the specs?",
                "Product details please",
                "Show product info",
            ],
            keywords=["product", "details", "specs", "information", "about"],
        ),
    ]

    print(f"  Defined {len(tools)} tools:")
    for tool in tools:
        print(f"    - {tool.name}: {len(tool.examples)} examples")

    # -------------------------------------------------------------------------
    # Step 2: Generate synthetic data (Rule-based)
    # -------------------------------------------------------------------------
    print("\nStep 2: Generating synthetic data (rule-based)...")

    generator = SyntheticDataGenerator(method="rule_based")

    # Generate with variations
    synthetic_data = generator.generate(
        tools=tools,
        num_variations=50,  # Variations per base pattern
        num_samples=1000,  # Target total samples
    )

    print(f"  Generated {len(synthetic_data)} samples")

    # Show some examples
    print("\n  Sample generated queries:")
    for i, sample in enumerate(synthetic_data[:10], 1):
        print(f"    {i:2d}. [{sample['tool']:20s}] {sample['query']}")

    # -------------------------------------------------------------------------
    # Step 3: Validate the generated data
    # -------------------------------------------------------------------------
    print("\nStep 3: Validating generated data...")

    validator = DataValidator()
    report = validator.validate(
        synthetic_data,
        min_samples_per_tool=50,
        warn_duplicates=True,
        warn_imbalance=True,
    )

    validator.print_report(report)

    # -------------------------------------------------------------------------
    # Step 4: Split into train/val/test with pattern groups
    # -------------------------------------------------------------------------
    print("\nStep 4: Splitting data (preserving pattern groups)...")

    splitter = PatternGroupSplitter()
    train_data, val_data, test_data = splitter.split(
        synthetic_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    # Verify no leakage
    print("\nStep 5: Checking for data leakage...")
    validator.check_leakage(train_data, test_data)

    # -------------------------------------------------------------------------
    # Step 6: Train the router
    # -------------------------------------------------------------------------
    print("\nStep 6: Training router with synthetic data...")

    router = FuncRoute()
    router.train(
        train_data=train_data,
        val_data=val_data,
        tools=tools,  # Pass tool definitions for better formatting
        config=TrainingConfig(
            output_dir="./synthetic_router",
            num_epochs=1,  # Increase for better results
            batch_size=4,
            save_steps=100,
        ),
    )

    # -------------------------------------------------------------------------
    # Step 7: Test the trained router
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Testing Trained Router")
    print("=" * 80)

    test_queries = [
        "Where is my package?",
        "Show me laptops under 500 dollars",
        "I want my money back",
        "What are the specifications of this item?",
        "Cancel my recent order",
        "Find wireless headphones",
    ]

    for query in test_queries:
        result = router.route(query)
        print(f"\nQuery: '{query}'")
        print(f"  → Tool: {result.tool}")
        print(f"  → Confidence: {result.confidence:.2%}")
        print(f"  → Latency: {result.latency_ms:.1f}ms")

        # Show alternatives if available
        if result.alternatives:
            print(f"  → Alternatives:")
            for alt_tool, alt_conf in result.alternatives[:3]:
                print(f"      - {alt_tool}: {alt_conf:.2%}")


def llm_based_example():
    """
    Example of LLM-based synthetic data generation.

    NOTE: This requires downloading a language model and is much slower
    than rule-based generation. Only use if you need more diverse variations.
    """
    print("\n" + "=" * 80)
    print("LLM-based Generation Example (Optional)")
    print("=" * 80)

    tools = [
        ToolDefinition(
            name="book_appointment",
            signature="book_appointment(date: str, service: str) -> dict",
            description="Book appointments for services",
            examples=[
                "Schedule a haircut for tomorrow",
                "Book a dentist appointment",
            ],
        ),
    ]

    # Use smaller model for faster generation
    generator = SyntheticDataGenerator(method="llm_based", llm_model="gpt2")

    print("\nGenerating with LLM (this may take a while)...")
    synthetic_data = generator.generate(
        tools=tools,
        num_samples=20,  # Keep small for demo
        domain_context="appointment booking for various services",
    )

    print(f"\nGenerated {len(synthetic_data)} LLM-based samples:")
    for i, sample in enumerate(synthetic_data[:10], 1):
        print(f"  {i:2d}. {sample['query']}")


if __name__ == "__main__":
    # Run main example
    main()

    # Uncomment to run LLM-based example (requires model download)
    # llm_based_example()
