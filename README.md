# FuncRoute

**Intelligent Function/Tool Routing using Fine-tuned FunctionGemma**

FuncRoute is a production-ready Python package for intelligent task routing in agentic AI systems. Fine-tune Google's FunctionGemma (270M parameters) to route user queries to the appropriate function or tool with high accuracy and low latency.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org/project/funcroute/)

## 🌟 Why FuncRoute?

**Problem:** Modern AI agents need to route user queries to the right tool/function among dozens of options. Traditional approaches using massive LLMs are:
- 💸 **Expensive** ($0.10+ per 1000 queries)
- 🐌 **Slow** (1-3 seconds per query)
- 🎯 **Inconsistent** (hallucinations, wrong tools)

**FuncRoute Solution:**
- 💰 **99% cheaper** (fine-tuned 270M model vs GPT-4)
- ⚡ **10-100x faster** (50-200ms per query)
- 🎯 **More accurate** (98%+ with proper training)
- 🔒 **Self-hosted** (no API costs, full control)

## 🚀 Features

### Core Capabilities
- ✅ **Easy Training**: Fine-tune FunctionGemma with your own data in minutes
- ✅ **Synthetic Data**: Generate 1000s of training samples automatically
- ✅ **Anti-Leakage**: Pattern group splitting prevents overfitting
- ✅ **Data Validation**: Automatic format checking and quality validation
- ✅ **Efficient Training**: LoRA + 4-bit quantization (runs on 8GB GPU)
- ✅ **Fast Inference**: Batch prediction, streaming, async support
- ✅ **Production Ready**: REST API, caching (10x speedup), monitoring

### Data & Training
- **Synthetic Data Generation**: Rule-based pattern expansion (like train.py)
- **Pattern Group Splitting**: Prevents data leakage between train/val/test
- **Data Validation**: Format checking, leakage detection, quality metrics
- **Flexible Input**: JSONL, CSV, pandas DataFrame, Hugging Face Datasets
- **Memory Efficient**: 4-bit quantization + LoRA (8GB GPU sufficient)

### Inference & Deployment
- **Batch Processing**: Parallel predictions with progress tracking
- **Streaming**: Process results as they arrive
- **Async Support**: Native asyncio for web frameworks
- **Caching**: LRU + TTL caching (5-10x speedup)
- **REST API**: FastAPI server with OpenAPI docs
- **CLI**: Complete command-line interface

### Evaluation & Monitoring
- **Metrics**: Accuracy, precision, recall, F1 per tool
- **Visualization**: Confusion matrices, performance charts
- **Cross-Validation**: K-fold validation support
- **Latency Tracking**: Per-query timing and statistics

## 📦 Installation

### From PyPI (Coming Soon)
```bash
pip install funcroute
```

### From Source
```bash
git clone https://github.com/yourusername/funcroute.git
cd funcroute
pip install -e .
```

### Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA GPU (recommended, 8GB+ VRAM)
- CPU supported but 10x slower

## 🎯 Quick Start

### 1. Simple Example (Complete Workflow)

```python
from funcroute import FuncRoute, TrainingConfig
from funcroute.core.config import ToolDefinition
from funcroute.data.generator import SyntheticDataGenerator
from funcroute.data.splitter import PatternGroupSplitter

# Step 1: Define your tools
tools = [
    ToolDefinition(
        name="manage_order",
        signature="manage_order(order_id: str) -> dict",
        description="Track and manage customer orders, check delivery status",
        examples=["Where is my order?", "Track package #12345"],
        keywords=["order", "track", "delivery", "shipping"],
    ),
    ToolDefinition(
        name="search_products",
        signature="search_products(query: str) -> list",
        description="Search for products in the catalog",
        examples=["Show me red dresses", "Find laptops under $1000"],
        keywords=["search", "find", "show", "products"],
    ),
    ToolDefinition(
        name="process_return",
        signature="process_return(order_id: str, reason: str) -> dict",
        description="Process product returns and refunds",
        examples=["Return this item", "I want a refund"],
        keywords=["return", "refund", "exchange"],
    ),
]

# Step 2: Generate synthetic training data
generator = SyntheticDataGenerator(method="rule_based")
data = generator.generate(
    tools=tools,
    num_variations=50,  # Creates 50 variations per pattern
    num_samples=5000,   # Target ~5000 total samples
)

# Step 3: Split with anti-leakage (prevents overfitting)
splitter = PatternGroupSplitter(seed=42)
train_data, val_data, test_data = splitter.split(
    data,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    verify_no_leakage=True,  # Automatic verification
)

# Step 4: Train the model
router = FuncRoute()
router.train(
    train_data=train_data,
    val_data=val_data,
    tools=tools,  # CRITICAL: Must provide tool definitions!
    config=TrainingConfig(
        output_dir="./my_router",
        num_epochs=3,           # 3 epochs for good accuracy
        batch_size=4,           # Adjust based on GPU memory
        learning_rate=2e-4,     # Standard for fine-tuning
        eval_strategy="epoch",  # Evaluate at end of each epoch
    ),
)

# Step 5: Save and load
# Model automatically saved to ./my_router with tool_definitions.json
loaded_router = FuncRoute.load("./my_router")

# Step 6: Make predictions
result = loaded_router.route("Where is my package?")
print(f"Tool: {result.tool}")           # manage_order
print(f"Confidence: {result.confidence:.1%}")  # 98.5%
print(f"Latency: {result.latency_ms:.1f}ms")   # 150ms
```

### 2. Using Pre-trained Model

```python
from funcroute import FuncRoute

# Load from Hugging Face Hub
router = FuncRoute.from_pretrained("scionoftech/functiongemma-e-commerce-tool-calling")

# Route queries
result = router.route("Where is my order?")
print(f"Tool: {result.tool}")  # manage_order
```

### 3. Production Deployment

```python
from funcroute.inference import Predictor, RouteCache

# Load trained model
router = FuncRoute.load("./my_router")

# Add caching for 10x speedup
cache = RouteCache(max_size=1000, ttl_seconds=3600)
predictor = Predictor(router, cache=cache)

# Batch prediction
queries = [
    "Where is my order?",
    "Show me laptops",
    "Return this item",
    # ... 100s more
]
results = predictor.predict_batch(queries, max_workers=4, show_progress=True)

# Async prediction (for web apps)
import asyncio
result = await predictor.predict_async("Where is my order?")
```

### 4. REST API Server

```bash
# Start server
funcroute serve --model ./my_router --port 8000

# Or in Python
python examples/server_example.py
```

```bash
# Make requests
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Where is my order?"}'

# Batch requests
curl -X POST http://localhost:8000/route/batch \
  -H "Content-Type: application/json" \
  -d '{"queries": ["Where is my order?", "Show me laptops"]}'

# Health check
curl http://localhost:8000/health
```

## 🛠️ CLI Usage

### Training
```bash
# Generate synthetic data
funcroute generate \
  --tools tools.json \
  --output synthetic.jsonl \
  --num-samples 5000

# Train model
funcroute train \
  --train-data train.jsonl \
  --val-data val.jsonl \
  --tools tools.json \
  --output-dir ./my_router \
  --num-epochs 3 \
  --batch-size 4

# Train with synthetic data generation
funcroute train \
  --tools tools.json \
  --output-dir ./my_router \
  --generate-data \
  --num-samples 5000
```

### Evaluation
```bash
# Evaluate model
funcroute evaluate \
  --model ./my_router \
  --test-data test.jsonl \
  --output metrics.json

# With visualizations
funcroute evaluate \
  --model ./my_router \
  --test-data test.jsonl \
  --plot \
  --output-dir ./eval_results
```

### Inference
```bash
# Single prediction
funcroute predict \
  --model ./my_router \
  --query "Where is my order?"

# Batch prediction
funcroute predict \
  --model ./my_router \
  --file queries.txt \
  --output results.jsonl

# Interactive mode
funcroute interactive --model ./my_router
```

### Serving
```bash
# Start REST API server
funcroute serve \
  --model ./my_router \
  --port 8000 \
  --cache-size 1000

# With custom host
funcroute serve \
  --model ./my_router \
  --host 0.0.0.0 \
  --port 8000
```

## 📊 Examples

We provide 9 comprehensive examples demonstrating all features:

1. **[simple_example.py](examples/simple_example.py)** - Complete workflow with 5000 samples
2. **[batch_prediction_example.py](examples/batch_prediction_example.py)** - 7 batch processing patterns
3. **[streaming_prediction_example.py](examples/streaming_prediction_example.py)** - 7 streaming patterns
4. **[async_prediction_example.py](examples/async_prediction_example.py)** - 9 async/await patterns
5. **[caching_example.py](examples/caching_example.py)** - 8 caching strategies
6. **[evaluation_example.py](examples/evaluation_example.py)** - Metrics and cross-validation
7. **[synthetic_data_example.py](examples/synthetic_data_example.py)** - Data generation
8. **[server_example.py](examples/server_example.py)** - REST API deployment
9. **[test_imports.py](examples/test_imports.py)** - Import verification

### Running Examples

```bash
# Complete workflow (recommended first example)
cd examples
python simple_example.py

# Batch processing (creates model for other examples)
python batch_prediction_example.py

# Then run dependent examples
python streaming_prediction_example.py
python async_prediction_example.py
python caching_example.py

# Standalone examples
python evaluation_example.py
python synthetic_data_example.py
python server_example.py
```

See [examples/README.md](examples/README.md) for detailed documentation.

## 🏗️ Architecture

```
funcroute/
├── funcroute/
│   ├── __init__.py              # Main exports
│   ├── cli.py                   # CLI interface
│   ├── core/
│   │   ├── config.py            # Configurations (TrainingConfig, ToolDefinition, etc.)
│   │   └── router.py            # FuncRoute main class
│   ├── training/
│   │   └── trainer.py           # Training orchestration
│   ├── inference/
│   │   ├── predictor.py         # Batch, streaming, async prediction
│   │   ├── cache.py             # LRU cache with TTL
│   │   └── server.py            # FastAPI REST server
│   ├── evaluation/
│   │   ├── evaluator.py         # Metrics computation
│   │   ├── metrics.py           # Metric functions
│   │   └── visualizer.py        # Plotting and charts
│   └── data/
│       ├── loader.py            # Data loading (JSONL, CSV, DataFrame)
│       ├── formatter.py         # FunctionGemma format conversion
│       ├── generator.py         # Synthetic data generation
│       ├── splitter.py          # Pattern group splitting
│       └── validator.py         # Data validation
├── examples/                    # 9 comprehensive examples
├── tests/                       # Test suite
└── docs/                        # Documentation
```

## 🎯 Data Format

### Training Data (JSONL)
```jsonl
{"query": "Where is my order?", "tool": "manage_order"}
{"query": "Show me red dresses", "tool": "search_products"}
{"query": "Return this item", "tool": "process_return"}
```

### Tool Definitions (Python)
```python
from funcroute.core.config import ToolDefinition

tools = [
    ToolDefinition(
        name="search_products",
        signature="search_products(query: str, category: str = None) -> list",
        description="Search for products in the catalog by name, category, or attributes",
        examples=[
            "Show me red dresses",
            "Find laptops under $1000",
            "Do you have iPhone 15?",
        ],
        keywords=["search", "find", "show", "looking", "browse"],
        parameters={
            "query": {"type": "string", "description": "Search query"},
            "category": {"type": "string", "description": "Product category", "required": False},
        },
    ),
]
```

### Tool Definitions (JSON)
```json
{
  "tools": [
    {
      "name": "search_products",
      "signature": "search_products(query: str) -> list",
      "description": "Search for products",
      "examples": ["Show me laptops", "Find red shoes"],
      "keywords": ["show", "find", "search"],
      "parameters": {
        "query": {"type": "string", "description": "Search query"}
      }
    }
  ]
}
```

## 🔧 Advanced Usage

### Pattern Group Splitting (Anti-Leakage)

**Problem:** Random splitting can leak pattern variations between train/test, causing inflated accuracy.

**Solution:** FuncRoute groups similar queries and splits by groups, not individual samples.

```python
from funcroute.data.splitter import PatternGroupSplitter

# Data with pattern groups (from SyntheticDataGenerator)
data = [
    {"query": "Where is my order?", "tool": "manage_order", "base_pattern": "order_status_1"},
    {"query": "Hi, where is my order?", "tool": "manage_order", "base_pattern": "order_status_1"},
    {"query": "Track my package", "tool": "manage_order", "base_pattern": "track_package_1"},
    # ...
]

splitter = PatternGroupSplitter(seed=42)
train, val, test = splitter.split(
    data,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    verify_no_leakage=True,  # Raises error if leakage detected
)

# Output:
# Pattern Group Splitting:
#   Total groups: 150
#   Train groups: 105 (3500 samples)
#   Val groups: 22 (750 samples)
#   Test groups: 23 (750 samples)
#  NO DATA LEAKAGE - Splits are clean!
```

### Data Validation

```python
from funcroute.data.validator import DataValidator

validator = DataValidator()

# Validate data quality
report = validator.validate(
    train_data,
    min_samples_per_tool=100,
    warn_duplicates=True,
    warn_imbalance=True,
)

if not report['is_valid']:
    print("Validation failed:")
    for error in report['errors']:
        print(f"  - {error}")
else:
    print(" Data is valid")

# Check for leakage
no_leakage = validator.check_leakage(train_data, test_data)
if not no_leakage:
    print(" Data leakage detected!")
```

### Caching Strategies

```python
from funcroute.inference import RouteCache, WarmupCache

# LRU cache with TTL
cache = RouteCache(max_size=1000, ttl_seconds=3600)
predictor = Predictor(router, cache=cache)

# Pre-warm cache with common queries
warmup = WarmupCache(predictor)
common_queries = ["Where is my order?", "Track package", "Return item"]
warmup.warmup(common_queries)

# Cache statistics
stats = cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
```

### Evaluation Metrics

```python
from funcroute.evaluation import Evaluator, Visualizer

# Evaluate on test set
evaluator = Evaluator(router, test_data)
metrics = evaluator.evaluate()

print(f"Overall Accuracy: {metrics['overall']['accuracy']:.2%}")
print(f"Per-tool metrics:")
for tool, tool_metrics in metrics['per_tool'].items():
    print(f"  {tool}: {tool_metrics['f1']:.2%} F1")

# Visualize results
visualizer = Visualizer(evaluator)
visualizer.plot_confusion_matrix(save_path="confusion.png")
visualizer.plot_per_tool_metrics(save_path="per_tool.png")
```


### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/funcroute.git
cd funcroute

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
black funcroute/
flake8 funcroute/
```

### Areas for Contribution

- 📝 Documentation improvements
- 🐛 Bug fixes and issue reports
- ✨ New examples and use cases
- 🔧 Performance optimizations
- 🌐 Framework integrations (LangChain, etc.)
- 📊 Visualization enhancements

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Google FunctionGemma**: Base model ([google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it))
- **Pre-trained Model**: E-commerce model by [functiongemma-e-commerce-tool-calling](https://huggingface.co/scionoftech/functiongemma-e-commerce-tool-calling)
- **Inspiration**: Based on train.py e-commerce routing example
- **Community**: Thanks to all contributors and users

**Upcoming v0.2:**
- 🔄 Streaming inference improvements
- 📊 Enhanced visualization dashboard
- 🌐 LangChain integration
- 🚀 Deployment guides (Docker, K8s)
- 📚 More examples and tutorials

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/scionoftech/funcroute/issues)
- **Discussions**: [GitHub Discussions](https://github.com/scionoftech/funcroute/discussions)

If you find FuncRoute useful, please consider starring the repository!