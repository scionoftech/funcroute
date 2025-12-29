# FuncRoute CLI Examples

Complete guide to using the FuncRoute command-line interface.

## Installation

```bash
# Basic installation
pip install funcroute

# With server support
pip install 'funcroute[server]'

# Development installation
pip install -e .
```

## Commands Overview

FuncRoute provides 5 main commands:
- `generate` - Generate synthetic training data
- `train` - Train a routing model
- `evaluate` - Evaluate model performance
- `predict` - Make predictions
- `serve` - Start REST API server

---

## 1. Generate Command

Generate synthetic training data from tool definitions.

### Basic Usage

```bash
# Generate data from tools file
funcroute generate \
    --tools examples/tools.json \
    --output data/training.jsonl \
    --num-samples 1000
```

### With All Options

```bash
funcroute generate \
    --tools examples/tools.json \
    --output data/training.jsonl \
    --num-samples 1000 \
    --num-variations 50 \
    --method rule_based \
    --seed 42
```

### Method Options

**Rule-based generation (default, recommended):**
```bash
funcroute generate \
    --tools tools.json \
    --output training.jsonl \
    --method rule_based \
    --num-variations 30 \
    --num-samples 500
```

**LLM-based generation (requires API key):**
```bash
# Set API key
export ANTHROPIC_API_KEY="your-key-here"

funcroute generate \
    --tools tools.json \
    --output training.jsonl \
    --method llm_based \
    --num-variations 20 \
    --num-samples 300
```

### Example Tools File Format

Create `tools.json`:
```json
[
  {
    "name": "manage_order",
    "signature": "manage_order(order_id: str) -> dict",
    "description": "Track and manage customer orders",
    "examples": [
      "Where is my order?",
      "Track my package",
      "Order status for #12345"
    ]
  },
  {
    "name": "search_products",
    "signature": "search_products(query: str) -> list",
    "description": "Search product catalog",
    "examples": [
      "Find laptops under $1000",
      "Show me wireless keyboards",
      "Search for monitors"
    ]
  },
  {
    "name": "process_return",
    "signature": "process_return(order_id: str) -> dict",
    "description": "Process returns and refunds",
    "examples": [
      "I want to return this",
      "Get a refund for order #123",
      "Return defective item"
    ]
  }
]
```

---

## 2. Train Command

Train a routing model from training data.

### Basic Usage

```bash
# Train with default settings
funcroute train \
    --data data/training.jsonl \
    --output models/my_router \
    --tools examples/tools.json
```

### With All Options

```bash
funcroute train \
    --data data/training.jsonl \
    --val-data data/validation.jsonl \
    --output models/my_router \
    --tools examples/tools.json \
    --model-name unsloth/Llama-3.2-1B-Instruct \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --max-seq-length 512 \
    --lora-rank 16 \
    --eval-steps 100 \
    --save-steps 200 \
    --seed 42
```

### Quick Training (for testing)

```bash
# Minimal epochs for quick iteration
funcroute train \
    --data data/training.jsonl \
    --output models/test_router \
    --tools tools.json \
    --epochs 1 \
    --batch-size 2
```

### Production Training

```bash
# Full training with evaluation
funcroute train \
    --data data/training.jsonl \
    --val-data data/validation.jsonl \
    --output models/production_router \
    --tools tools.json \
    --epochs 5 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --eval-steps 50 \
    --save-steps 100
```

### Training Data Format

Each line in `training.jsonl` should be:
```json
{
  "query": "Where is my order #12345?",
  "tool": "manage_order"
}
```

---

## 3. Evaluate Command

Evaluate a trained model's performance.

### Basic Evaluation

```bash
# Evaluate with test data
funcroute evaluate \
    --model models/my_router \
    --data data/test.jsonl
```

### With Visualization

```bash
# Generate visualizations
funcroute evaluate \
    --model models/my_router \
    --data data/test.jsonl \
    --visualize \
    --output-dir evaluation_results
```

This creates:
- `confusion_matrix.png` - Confusion matrix heatmap
- `per_tool_metrics.png` - Bar chart of per-tool performance
- `confidence_distribution.png` - Confidence score distribution
- `metrics.json` - Detailed metrics in JSON format

### Custom Visualization Options

```bash
funcroute evaluate \
    --model models/my_router \
    --data data/test.jsonl \
    --visualize \
    --output-dir results \
    --confidence-threshold 0.8
```

### Reading Evaluation Results

The command outputs:
```
Evaluation Results:
  Accuracy: 95.2%
  Precision: 94.8%
  Recall: 95.1%
  F1 Score: 94.9%

Per-Tool Metrics:
  manage_order    - F1: 96.1% (precision: 95.8%, recall: 96.4%)
  search_products - F1: 94.2% (precision: 93.9%, recall: 94.5%)
  process_return  - F1: 94.5% (precision: 94.7%, recall: 94.3%)
```

---

## 4. Predict Command

Make predictions with a trained model.

### Single Query

```bash
# Predict single query
funcroute predict \
    --model models/my_router \
    --query "Where is my order?"
```

Output:
```
Query: Where is my order?
Tool: manage_order
Confidence: 98.5%
Latency: 45.2ms
```

### With Alternatives

```bash
# Show alternative tools
funcroute predict \
    --model models/my_router \
    --query "Where is my order?" \
    --alternatives
```

Output:
```
Query: Where is my order?

Primary Prediction:
  Tool: manage_order
  Confidence: 98.5%

Alternatives:
  1. track_package    - 78.2%
  2. order_status     - 65.3%
  3. search_products  - 12.1%
```

### Batch Prediction from File

```bash
# Predict from file (one query per line)
funcroute predict \
    --model models/my_router \
    --file queries.txt \
    --output predictions.jsonl
```

**queries.txt:**
```
Where is my order?
Find laptops under $1000
I want to return this
Track my package
```

**predictions.jsonl:**
```json
{"query": "Where is my order?", "tool": "manage_order", "confidence": 0.985}
{"query": "Find laptops under $1000", "tool": "search_products", "confidence": 0.972}
{"query": "I want to return this", "tool": "process_return", "confidence": 0.968}
{"query": "Track my package", "tool": "manage_order", "confidence": 0.991}
```

### With Caching

```bash
# Enable caching for repeated queries
funcroute predict \
    --model models/my_router \
    --file queries.txt \
    --cache-size 1000
```

---

## 5. Serve Command

Start a REST API server.

### Basic Server

```bash
# Start server on default port 8000
funcroute serve --model models/my_router
```

### Custom Host and Port

```bash
# Custom host and port
funcroute serve \
    --model models/my_router \
    --host 0.0.0.0 \
    --port 5000
```

### With Caching

```bash
# Enable caching for better performance
funcroute serve \
    --model models/my_router \
    --port 8000 \
    --cache-size 10000 \
    --cache-ttl 300
```

Parameters:
- `--cache-size`: Maximum number of cached predictions (default: 1000)
- `--cache-ttl`: Cache entry TTL in seconds (default: None = no expiration)

### Production Server

```bash
# Production settings
funcroute serve \
    --model models/production_router \
    --host 0.0.0.0 \
    --port 8000 \
    --cache-size 50000 \
    --cache-ttl 600 \
    --workers 4
```

### Testing the Server

Once started, test with:

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Where is my order?"}'

# Batch prediction
curl -X POST http://localhost:8000/route/batch \
  -H "Content-Type: application/json" \
  -d '{"queries": ["Where is my order?", "Find laptops"]}'

# Cache stats
curl http://localhost:8000/cache/stats

# Server stats
curl http://localhost:8000/stats
```

### OpenAPI Documentation

When server is running, visit:
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Complete Workflows

### Workflow 1: From Scratch to Production

```bash
# 1. Generate training data
funcroute generate \
    --tools tools.json \
    --output data/full_dataset.jsonl \
    --num-samples 1000

# 2. Split data manually or use Python script
# (training: 70%, validation: 15%, test: 15%)

# 3. Train model
funcroute train \
    --data data/train.jsonl \
    --val-data data/val.jsonl \
    --output models/production \
    --tools tools.json \
    --epochs 3

# 4. Evaluate model
funcroute evaluate \
    --model models/production \
    --data data/test.jsonl \
    --visualize \
    --output-dir evaluation

# 5. Deploy server
funcroute serve \
    --model models/production \
    --port 8000 \
    --cache-size 10000
```

### Workflow 2: Quick Iteration

```bash
# Generate small dataset
funcroute generate \
    --tools tools.json \
    --output data/quick.jsonl \
    --num-samples 100

# Quick train
funcroute train \
    --data data/quick.jsonl \
    --output models/test \
    --tools tools.json \
    --epochs 1

# Test prediction
funcroute predict \
    --model models/test \
    --query "test query"
```

### Workflow 3: Model Comparison

```bash
# Train model A
funcroute train \
    --data data/train.jsonl \
    --output models/model_a \
    --tools tools.json \
    --epochs 3

# Train model B with different params
funcroute train \
    --data data/train.jsonl \
    --output models/model_b \
    --tools tools.json \
    --epochs 5 \
    --learning-rate 3e-4

# Evaluate both
funcroute evaluate \
    --model models/model_a \
    --data data/test.jsonl \
    --visualize \
    --output-dir eval_a

funcroute evaluate \
    --model models/model_b \
    --data data/test.jsonl \
    --visualize \
    --output-dir eval_b

# Compare metrics in eval_a/metrics.json vs eval_b/metrics.json
```

---

## Environment Variables

```bash
# API key for LLM-based generation
export ANTHROPIC_API_KEY="your-key-here"

# Custom cache directory
export FUNCROUTE_CACHE_DIR="/path/to/cache"

# Logging level
export FUNCROUTE_LOG_LEVEL="DEBUG"
```

---

## Common Issues and Solutions

### Issue: Out of Memory During Training

**Solution:** Reduce batch size
```bash
funcroute train \
    --data train.jsonl \
    --output model \
    --tools tools.json \
    --batch-size 1  # Reduce from default 4
```

### Issue: Slow Predictions

**Solution:** Enable caching
```bash
funcroute serve \
    --model model \
    --cache-size 10000 \
    --cache-ttl 300
```

### Issue: Low Accuracy

**Solutions:**
1. Generate more training data
2. Increase training epochs
3. Improve tool examples in tools.json
4. Add more variations to data generation

```bash
# More data
funcroute generate \
    --tools tools.json \
    --output data.jsonl \
    --num-samples 2000 \
    --num-variations 50

# More epochs
funcroute train \
    --data data.jsonl \
    --output model \
    --tools tools.json \
    --epochs 5
```

### Issue: Server Won't Start

**Check:**
1. Is port already in use?
2. Is FastAPI installed? `pip install 'funcroute[server]'`
3. Try different port: `--port 8001`

---

## Advanced Usage

### Custom Model

```bash
funcroute train \
    --data train.jsonl \
    --output model \
    --tools tools.json \
    --model-name "meta-llama/Llama-3-8B-Instruct"
```

### Distributed Training (Future)

```bash
# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 funcroute train \
    --data train.jsonl \
    --output model \
    --tools tools.json \
    --distributed
```

### Custom Evaluation Metrics

```bash
funcroute evaluate \
    --model model \
    --data test.jsonl \
    --metrics accuracy precision recall f1 \
    --confidence-threshold 0.9
```

---

## Tips and Best Practices

### Data Generation
- Start with rule-based (faster, more consistent)
- Use 30-50 variations per tool
- Aim for 500-1000 samples minimum
- Include diverse examples in tool definitions

### Training
- Use validation data to prevent overfitting
- Start with 3 epochs, increase if accuracy improves
- Batch size of 4 is good for most cases
- Save checkpoints with `--save-steps`

### Evaluation
- Always use held-out test data
- Generate visualizations to understand errors
- Check per-tool metrics for imbalanced performance
- Confidence threshold of 0.8-0.9 is typical

### Prediction
- Use caching for production deployments
- Monitor cache hit rates (aim for >80%)
- Set appropriate TTL based on model update frequency
- Use batch predictions for bulk operations

### Serving
- Enable CORS if needed for web apps
- Use reverse proxy (nginx) for production
- Set cache size based on available memory
- Monitor server stats endpoint regularly

---

## Complete Example Session

```bash
# Setup
mkdir -p funcroute_project/{data,models,evaluation}
cd funcroute_project

# Create tools definition
cat > tools.json << 'EOF'
[
  {
    "name": "manage_order",
    "signature": "manage_order(order_id: str) -> dict",
    "description": "Track and manage customer orders",
    "examples": ["Where is my order?", "Track package #123"]
  },
  {
    "name": "search_products",
    "signature": "search_products(query: str) -> list",
    "description": "Search product catalog",
    "examples": ["Find laptops", "Show me keyboards"]
  }
]
EOF

# Generate data
echo "Generating training data..."
funcroute generate \
    --tools tools.json \
    --output data/all_data.jsonl \
    --num-samples 600

# Split data (70/15/15)
head -n 420 data/all_data.jsonl > data/train.jsonl
tail -n +421 data/all_data.jsonl | head -n 90 > data/val.jsonl
tail -n 90 data/all_data.jsonl > data/test.jsonl

# Train
echo "Training model..."
funcroute train \
    --data data/train.jsonl \
    --val-data data/val.jsonl \
    --output models/my_router \
    --tools tools.json \
    --epochs 3

# Evaluate
echo "Evaluating model..."
funcroute evaluate \
    --model models/my_router \
    --data data/test.jsonl \
    --visualize \
    --output-dir evaluation

# Test prediction
echo "Testing prediction..."
funcroute predict \
    --model models/my_router \
    --query "Where is my order #12345?" \
    --alternatives

# Start server
echo "Starting server..."
funcroute serve \
    --model models/my_router \
    --port 8000 \
    --cache-size 1000

# In another terminal, test the API:
# curl http://localhost:8000/health
# curl -X POST http://localhost:8000/route -H "Content-Type: application/json" -d '{"query": "Where is my order?"}'
```

---

## Getting Help

```bash
# General help
funcroute --help

# Command-specific help
funcroute generate --help
funcroute train --help
funcroute evaluate --help
funcroute predict --help
funcroute serve --help
```

For more examples, see the [examples/](../examples/) directory.
