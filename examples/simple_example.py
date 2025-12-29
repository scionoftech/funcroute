"""
Simple FuncRoute Example

Demonstrates complete workflow:
1. Generate synthetic training data (5000 samples)
2. Split with pattern group anti-leakage
3. Train and validate model
4. Save trained model
5. Load model and make predictions

This follows all best practices from train.py.
"""

from funcroute import FuncRoute, TrainingConfig
from funcroute.core.config import ToolDefinition
from funcroute.data.generator import SyntheticDataGenerator
from funcroute.data.splitter import PatternGroupSplitter
from funcroute.data.validator import DataValidator


def main():
    print("=" * 80)
    print("FuncRoute Simple Example - Complete Workflow")
    print("=" * 80)

    # =========================================================================
    # Step 1: Define Tools
    # =========================================================================
    print("\n[Step 1/6] Defining tools...")

    tools = [
        ToolDefinition(
            name="manage_order",
            signature="manage_order(order_id: str) -> dict",
            description="Track and manage customer orders, check delivery status, and update shipping information",
            examples=[
                "Where is my order?",
                "Track my package",
                "Order status for #12345",
                "When will my order arrive?",
                "Update shipping address",
            ],
            keywords=["order", "track", "package", "delivery", "shipment", "shipping"],
        ),
        ToolDefinition(
            name="search_products",
            signature="search_products(query: str) -> list",
            description="Search for products in the catalog by name, category, or attributes",
            examples=[
                "Show me red dresses",
                "Find laptops under $1000",
                "Do you have iPhone 15?",
                "Looking for running shoes",
                "Show new arrivals",
            ],
            keywords=["search", "find", "show", "looking", "browse", "products"],
        ),
        ToolDefinition(
            name="process_return",
            signature="process_return(order_id: str, reason: str) -> dict",
            description="Process product returns, refunds, and exchanges",
            examples=[
                "I want to return this",
                "Get a refund for my order",
                "Wrong item received",
                "Item is damaged",
                "Exchange for different size",
            ],
            keywords=["return", "refund", "exchange", "damaged", "wrong", "defective"],
        ),
    ]

    print(f"✅ Defined {len(tools)} tools:")
    for tool in tools:
        print(f"   - {tool.name}")

    # =========================================================================
    # Step 2: Generate Synthetic Training Data
    # =========================================================================
    print("\n[Step 2/6] Generating synthetic training data...")
    print("   Target: ~5000 samples with pattern groups")

    generator = SyntheticDataGenerator(method="rule_based")

    # Generate data with pattern groups
    # num_variations controls pattern diversity
    # num_samples is total target (actual may vary based on patterns)
    data = generator.generate(
        tools=tools,
        num_variations=50,  # 50 variations per pattern
        num_samples=5000,   # Target ~5000 samples
    )

    print(f"✅ Generated {len(data)} training samples")

    # Show sample data
    print(f"\n   Sample data:")
    for i, sample in enumerate(data[:3], 1):
        print(f"   {i}. '{sample['query']}' → {sample['tool']}")

    # =========================================================================
    # Step 3: Split Data (Pattern Group Anti-Leakage)
    # =========================================================================
    print("\n[Step 3/6] Splitting data with pattern group anti-leakage...")

    splitter = PatternGroupSplitter(seed=42)
    train_data, val_data, test_data = splitter.split(
        data,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        verify_no_leakage=True,  # Automatic leakage check
    )

    print(f"\n✅ Data split complete:")
    print(f"   Training:   {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
    print(f"   Validation: {len(val_data)} samples ({len(val_data)/len(data)*100:.1f}%)")
    print(f"   Test:       {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")

    # =========================================================================
    # Step 4: Validate Data (Optional but Recommended)
    # =========================================================================
    print("\n[Step 4/6] Validating data quality...")

    validator = DataValidator()

    # Validate data quality
    print("   Checking data quality...")
    report = validator.validate(train_data, min_samples_per_tool=100)
    if not report['is_valid']:
        print("   ❌ Data validation failed:")
        for error in report['errors']:
            print(f"      - {error}")
        raise ValueError("Data validation failed!")
    print("   ✅ Data quality is good")

    # Check for data leakage (already done by splitter, but good to verify)
    print("   Double-checking for data leakage...")
    no_leakage = validator.check_leakage(train_data, test_data)
    if not no_leakage:
        raise ValueError("❌ Data leakage detected!")

    # =========================================================================
    # Step 5: Train Model
    # =========================================================================
    print("\n[Step 5/6] Training routing model...")
    print("   This may take 10-20 minutes depending on your hardware...")

    router = FuncRoute()

    router.train(
        train_data=train_data,
        val_data=val_data,
        tools=tools,  # CRITICAL: Must provide tools!
        config=TrainingConfig(
            output_dir="./simple_router",
            num_epochs=3,           # 3 epochs for good accuracy
            batch_size=4,           # Adjust based on GPU memory
            learning_rate=2e-4,     # Standard for fine-tuning
            save_steps=100,         # Save checkpoints every 100 steps
            eval_strategy="epoch",  # Evaluate at end of each epoch
            logging_steps=10,       # Log every 10 steps
            warmup_ratio=0.1,       # 10% warmup
            save_total_limit=2,     # Keep only 2 checkpoints
        ),
    )

    print("\n✅ Training complete!")
    print(f"   Model saved to: ./simple_router")

    # =========================================================================
    # Step 6: Test Trained Model
    # =========================================================================
    print("\n[Step 6/6] Testing trained model...")

    test_queries = [
        "Where is my package?",
        "Show me laptops under $800",
        "I want my money back",
        "Track order #12345",
        "Find wireless keyboards",
        "Return defective item",
        "When will my order arrive?",
        "Do you have iPhone cases?",
        "Exchange for different color",
        "Search for running shoes",
    ]

    print(f"\n📊 Testing with {len(test_queries)} queries:\n")
    print(f"{'Query':<40s} {'Tool':<20s} {'Confidence':>12s}")
    print("-" * 80)

    correct = 0
    expected_tools = {
        "Where is my package?": "manage_order",
        "Show me laptops under $800": "search_products",
        "I want my money back": "process_return",
        "Track order #12345": "manage_order",
        "Find wireless keyboards": "search_products",
        "Return defective item": "process_return",
        "When will my order arrive?": "manage_order",
        "Do you have iPhone cases?": "search_products",
        "Exchange for different color": "process_return",
        "Search for running shoes": "search_products",
    }

    for query in test_queries:
        result = router.route(query)
        expected = expected_tools.get(query, "unknown")
        is_correct = result.tool == expected

        if is_correct:
            correct += 1
            marker = "✅"
        else:
            marker = "❌"

        print(f"{query:<40s} {result.tool:<20s} {result.confidence:>11.1%} {marker}")

    accuracy = correct / len(test_queries) * 100
    print("-" * 80)
    print(f"\n📈 Accuracy: {correct}/{len(test_queries)} ({accuracy:.1f}%)")

    # =========================================================================
    # Bonus: Load Model and Re-test
    # =========================================================================
    print("\n" + "=" * 80)
    print("Bonus: Loading Saved Model")
    print("=" * 80)

    print("\nLoading model from disk...")
    loaded_router = FuncRoute.load("./simple_router")
    print("✅ Model loaded successfully!")

    print("\nTesting loaded model with new queries...")

    new_queries = [
        "Check my order status",
        "Looking for blue jeans",
        "I want a refund",
    ]

    print(f"\n{'Query':<40s} {'Tool':<20s} {'Confidence':>12s}")
    print("-" * 80)

    for query in new_queries:
        result = loaded_router.route(query)
        print(f"{query:<40s} {result.tool:<20s} {result.confidence:>11.1%}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("✅ Example Complete!")
    print("=" * 80)

    print(f"\nWhat we did:")
    print(f"  1. ✅ Defined 3 tools with proper metadata")
    print(f"  2. ✅ Generated {len(data)} synthetic samples with pattern groups")
    print(f"  3. ✅ Split data with anti-leakage (70/15/15)")
    print(f"  4. ✅ Validated data quality and checked for leakage")
    print(f"  5. ✅ Trained model for 3 epochs")
    print(f"  6. ✅ Saved model to ./simple_router")
    print(f"  7. ✅ Loaded model and tested predictions")

    print(f"\nResults:")
    print(f"  📊 Test Accuracy: {accuracy:.1f}%")
    print(f"  💾 Model Location: ./simple_router")
    print(f"  📈 Training Samples: {len(train_data)}")
    print(f"  📉 Validation Samples: {len(val_data)}")
    print(f"  🧪 Test Samples: {len(test_data)}")

    print(f"\nNext steps:")
    print(f"  - Load model: FuncRoute.load('./simple_router')")
    print(f"  - Make predictions: router.route('your query')")
    print(f"  - Deploy as API: See examples/server_example.py")
    print(f"  - Use CLI: funcroute predict --model ./simple_router --query 'test'")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
