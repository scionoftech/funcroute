# Simple Example Guide

Complete guide to running the updated simple_example.py with 5000 training samples.

## What This Example Does

The updated `simple_example.py` demonstrates the complete FuncRoute workflow following all best practices from train.py:

1. ✅ Define tools with proper metadata
2. ✅ Generate 5000 synthetic training samples
3. ✅ Split data with pattern group anti-leakage
4. ✅ Validate data format and check for leakage
5. ✅ Train model for 3 epochs
6. ✅ Save trained model to disk
7. ✅ Load model back and test predictions

## Quick Start

```bash
cd examples
python simple_example.py
```

## What to Expect

### Runtime
- **With GPU:** 10-20 minutes
- **With CPU:** 1-2 hours (not recommended)

### Output Structure

The example produces clear, step-by-step output:

```
================================================================================
FuncRoute Simple Example - Complete Workflow
================================================================================

[Step 1/6] Defining tools...
✅ Defined 3 tools:
   - manage_order
   - search_products
   - process_return

[Step 2/6] Generating synthetic training data...
   Target: ~5000 samples with pattern groups
✅ Generated 5000 training samples

   Sample data:
   1. 'Where is my order?' → manage_order
   2. 'Show me red dresses' → search_products
   3. 'I want to return this' → process_return

[Step 3/6] Splitting data with pattern group anti-leakage...

Pattern Group Splitting:
  Total groups: 150
  Total samples: 5000
  Train groups: 105
  Val groups: 22
  Test groups: 23

Expanded to samples:
  Train: 3500
  Val: 750
  Test: 750

================================================================================
DATA LEAKAGE CHECK
================================================================================
Train-Val overlap: 0 queries
Train-Test overlap: 0 queries
Val-Test overlap: 0 queries
✅ NO DATA LEAKAGE - Splits are clean!
================================================================================

✅ Data split complete:
   Training:   3500 samples (70.0%)
   Validation: 750 samples (15.0%)
   Test:       750 samples (15.0%)

[Step 4/6] Validating data quality...
   Checking data format...
   ✅ Data format is valid
   Double-checking for data leakage...

✅ NO DATA LEAKAGE
   Train: 3500 unique queries
   Test: 750 unique queries
   Overlap: 0

[Step 5/6] Training routing model...
   This may take 10-20 minutes depending on your hardware...

================================================================================
FUNCROUTE TRAINING
================================================================================

[1/6] Loading training data...
✅ Loaded 3500 training samples
✅ Loaded 750 validation samples

[2/6] Preparing tool definitions...
✅ Using 3 provided tool definitions

[3/6] Formatting data for FunctionGemma...
✅ Formatted 3500 training samples
✅ Formatted 750 validation samples

[4/6] Loading base model...
✅ Loaded model: google/functiongemma-270m-it

[5/6] Configuring training...
✅ Training configuration ready

[6/6] Training model...
[Training progress bars and metrics...]

✅ Training complete!
   Model saved to: ./simple_router

[Step 6/6] Testing trained model...

📊 Testing with 10 queries:

Query                                    Tool                 Confidence
--------------------------------------------------------------------------------
Where is my package?                     manage_order              98.5% ✅
Show me laptops under $800               search_products           97.2% ✅
I want my money back                     process_return            96.8% ✅
Track order #12345                       manage_order              99.1% ✅
Find wireless keyboards                  search_products           98.3% ✅
Return defective item                    process_return            97.5% ✅
When will my order arrive?               manage_order              98.9% ✅
Do you have iPhone cases?                search_products           96.7% ✅
Exchange for different color             process_return            95.4% ✅
Search for running shoes                 search_products           97.8% ✅
--------------------------------------------------------------------------------

📈 Accuracy: 10/10 (100.0%)

================================================================================
Bonus: Loading Saved Model
================================================================================

Loading model from disk...
✅ Model loaded successfully!

Testing loaded model with new queries...

Query                                    Tool                 Confidence
--------------------------------------------------------------------------------
Check my order status                    manage_order              98.2%
Looking for blue jeans                   search_products           97.5%
I want a refund                          process_return            96.9%

================================================================================
✅ Example Complete!
================================================================================

What we did:
  1. ✅ Defined 3 tools with proper metadata
  2. ✅ Generated 5000 synthetic samples with pattern groups
  3. ✅ Split data with anti-leakage (70/15/15)
  4. ✅ Validated data format and checked for leakage
  5. ✅ Validated data format and checked for leakage
  6. ✅ Trained model for 3 epochs
  7. ✅ Saved model to ./simple_router
  8. ✅ Loaded model and tested predictions

Results:
  📊 Test Accuracy: 100.0%
  💾 Model Location: ./simple_router
  📈 Training Samples: 3500
  📉 Validation Samples: 750
  🧪 Test Samples: 750

Next steps:
  - Load model: FuncRoute.load('./simple_router')
  - Make predictions: router.route('your query')
  - Deploy as API: See examples/server_example.py
  - Use CLI: funcroute predict --model ./simple_router --query 'test'

================================================================================
```

## Key Features

### 1. Synthetic Data Generation (5000 samples)

```python
generator = SyntheticDataGenerator(method="rule_based")
data = generator.generate(
    tools=tools,
    num_variations=50,  # Creates 50 variations per pattern
    num_samples=5000,   # Target ~5000 total samples
)
```

**Benefits:**
- Automatic pattern group creation
- Diverse query variations
- Balanced across tools
- Includes base_pattern field for anti-leakage

### 2. Pattern Group Anti-Leakage Splitting

```python
splitter = PatternGroupSplitter(seed=42)
train_data, val_data, test_data = splitter.split(
    data,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    verify_no_leakage=True,  # Automatic verification
)
```

**Prevents:**
- Pattern memorization
- Inflated validation scores
- Data leakage between splits

### 3. Complete Training Pipeline

```python
router.train(
    train_data=train_data,
    val_data=val_data,
    tools=tools,  # Proper tool definitions
    config=TrainingConfig(
        output_dir="./simple_router",
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,
        save_steps=100,
        eval_steps=50,
    ),
)
```

**Features:**
- Full tool metadata
- FunctionGemma format validation
- Checkpointing
- Validation during training

### 4. Model Persistence

```python
# Save (automatic during training)
router.train(..., config=TrainingConfig(output_dir="./simple_router"))

# Load
loaded_router = FuncRoute.load("./simple_router")

# Use
result = loaded_router.route("test query")
```

**Saved Files:**
```
simple_router/
├── config.json              # Model configuration
├── tokenizer_config.json    # Tokenizer settings
├── adapter_config.json      # LoRA adapter config
├── adapter_model.safetensors # LoRA weights
└── tools.json               # Tool definitions
```

## Requirements

### Hardware
- **Minimum:** 8GB RAM, CPU (slow)
- **Recommended:** 16GB RAM + GPU with 8GB VRAM
- **Optimal:** 32GB RAM + GPU with 16GB VRAM

### Software
```bash
# Core dependencies (auto-installed with package)
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
trl>=0.7.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
datasets>=2.14.0
scikit-learn>=1.3.0
```

## Customization

### Adjust Data Size

```python
# Generate more data
data = generator.generate(
    tools=tools,
    num_variations=100,  # More variations
    num_samples=10000,   # More samples
)
```

### Adjust Training

```python
# Faster training (less accuracy)
config=TrainingConfig(
    num_epochs=1,      # Fewer epochs
    batch_size=8,      # Larger batches
)

# Better accuracy (slower)
config=TrainingConfig(
    num_epochs=5,      # More epochs
    batch_size=2,      # Smaller batches (more updates)
    learning_rate=1e-4, # Lower LR
)
```

### Add More Tools

```python
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
        examples=["Return item", "Get refund"],
    ),
    # Add your new tools here
    ToolDefinition(
        name="manage_account",
        signature="manage_account(action: str) -> dict",
        description="Manage user account settings",
        examples=["Update email", "Change password"],
    ),
]
```

## Troubleshooting

### Out of Memory

**Error:** `CUDA out of memory`

**Solution:** Reduce batch size
```python
config=TrainingConfig(
    batch_size=1,  # Smallest possible
    gradient_accumulation_steps=4,  # Maintain effective batch size
)
```

### Slow Training

**Issue:** Training takes hours

**Solutions:**
1. Enable GPU: Install CUDA-enabled PyTorch
2. Reduce data: `num_samples=1000`
3. Reduce epochs: `num_epochs=1`

### Low Accuracy

**Issue:** Test accuracy < 90%

**Solutions:**
1. More data: `num_samples=10000`
2. More epochs: `num_epochs=5`
3. More variations: `num_variations=100`
4. Better tool descriptions and examples

## Files Created

After running the example:

```
examples/
├── simple_example.py        # This example
└── simple_router/           # Trained model (created)
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── config.json
    ├── tokenizer_config.json
    └── tools.json
```

## Next Steps

### 1. Use the Trained Model

```python
from funcroute import FuncRoute

# Load
router = FuncRoute.load("./simple_router")

# Predict
result = router.route("Where is my package?")
print(f"Tool: {result.tool}")
print(f"Confidence: {result.confidence:.1%}")
```

### 2. Deploy as REST API

```bash
# See server_example.py
funcroute serve --model ./simple_router --port 8000
```

### 3. Use CLI

```bash
# Single prediction
funcroute predict --model ./simple_router --query "test query"

# Batch prediction
funcroute predict --model ./simple_router --file queries.txt
```

### 4. Evaluate on Test Set

```python
from funcroute import FuncRoute
from funcroute.evaluation import Evaluator, Visualizer

router = FuncRoute.load("./simple_router")
evaluator = Evaluator(router, test_data)
metrics = evaluator.evaluate()

print(f"Accuracy: {metrics['overall']['accuracy']:.1%}")

# Visualize
visualizer = Visualizer(evaluator)
visualizer.plot_confusion_matrix()
visualizer.plot_per_tool_metrics()
```

## Comparison to Old Version

### Old simple_example.py ❌
- 15 training samples (too few)
- Manual train/val split (data leakage)
- No tool definitions (auto-extraction)
- No validation
- No model saving/loading

### New simple_example.py ✅
- 5000 training samples (sufficient)
- Pattern group splitting (no leakage)
- Proper tool definitions
- Format and leakage validation
- Model save and load
- Follows all train.py best practices

## Summary

This example demonstrates **production-ready** FuncRoute usage:

✅ **Data Generation:** Synthetic data with pattern groups
✅ **Anti-Leakage:** Pattern group splitting
✅ **Validation:** Format and leakage checks
✅ **Training:** Full pipeline with checkpointing
✅ **Persistence:** Save and load models
✅ **Testing:** Accuracy measurement
✅ **Best Practices:** Follows train.py standards

Perfect for understanding the complete workflow before building your own application!
