# FuncRoute Examples

Comprehensive examples demonstrating all FuncRoute features.

## Quick Start

All examples require a trained model. Run the batch example first to train a model:

```bash
cd examples
python batch_prediction_example.py
```

This creates `batch_example_model/` used by all other examples.

## Example Files

### 1. Basic Examples

#### [basic_example.py](basic_example.py)
**What it shows:** Complete workflow from data generation to prediction
- Generate synthetic training data
- Train a routing model
- Make predictions
- Evaluate performance

**Run it:**
```bash
python examples/basic_example.py
```

#### [server_example.py](server_example.py)
**What it shows:** REST API server deployment
- Basic server setup
- Server with caching
- Custom endpoints
- API testing
- Curl command examples

**Run it:**
```bash
# Run specific example
python examples/server_example.py 1  # Basic server
python examples/server_example.py 2  # Server with cache
python examples/server_example.py 3  # Custom app
python examples/server_example.py 4  # Test API (server must be running)
python examples/server_example.py 5  # Show curl commands

# Run all examples
python examples/server_example.py
```

### 2. Prediction Patterns

#### [batch_prediction_example.py](batch_prediction_example.py)
**What it shows:** Efficient batch prediction patterns
- Basic batch prediction with progress tracking
- Controlling parallelization (1, 2, 4, 8 workers)
- Filtering results by confidence
- Real-time callbacks
- Error handling without batch failure
- Performance comparison (sequential vs batch)
- Detailed progress tracking

**7 Examples:**
1. Basic batch prediction
2. Controlling parallelization
3. Batch with filtering
4. Batch with callbacks
5. Error handling
6. Performance comparison
7. Progress tracking

**Run it:**
```bash
python examples/batch_prediction_example.py     # All examples
python examples/batch_prediction_example.py 1   # Specific example
python examples/batch_prediction_example.py 2
# ... etc
```

**Key Features:**
- ThreadPoolExecutor for parallel processing
- Order preservation
- Progress bars with tqdm
- Graceful error handling
- Speedup measurements

#### [streaming_prediction_example.py](streaming_prediction_example.py)
**What it shows:** Streaming prediction for real-time processing
- Basic streaming (unordered results)
- Early processing (start before all complete)
- Streaming vs batch comparison
- Early termination with `predict_until()`
- Multi-stage processing pipeline
- Real-time aggregation
- Streaming to database

**7 Examples:**
1. Basic streaming prediction
2. Early processing with streaming
3. Streaming vs batch comparison
4. Streaming with early termination
5. Multi-stage streaming pipeline
6. Real-time aggregation
7. Streaming to database

**Run it:**
```bash
python examples/streaming_prediction_example.py     # All examples
python examples/streaming_prediction_example.py 4   # Early termination example
```

**Key Features:**
- Unordered result processing
- Time-to-first-result optimization
- Conditional termination
- Pipeline patterns
- Incremental database saves

#### [async_prediction_example.py](async_prediction_example.py)
**What it shows:** Asynchronous prediction with asyncio
- Basic async prediction
- Concurrent predictions with `asyncio.gather()`
- Async batch prediction
- Async with caching
- FastAPI integration patterns
- Async streaming with generators
- Timeout handling
- Event loop integration
- Production async patterns

**9 Examples:**
1. Basic async prediction
2. Concurrent async predictions
3. Async batch prediction
4. Async with cache
5. Async web integration (FastAPI pattern)
6. Async streaming
7. Async with timeout
8. Event loop integration
9. Production async patterns

**Run it:**
```bash
python examples/async_prediction_example.py     # All examples
python examples/async_prediction_example.py 5   # FastAPI integration
python examples/async_prediction_example.py 9   # Production patterns
```

**Key Features:**
- `asyncio.gather()` for concurrency
- Rate limiting with semaphores
- Timeout handling
- Error resilience
- Background task queues
- Web framework integration

### 3. Performance Optimization

#### [caching_example.py](caching_example.py)
**What it shows:** Caching for performance optimization
- Basic cache usage (hits/misses)
- Performance impact measurement
- LRU eviction behavior
- TTL expiration
- Cache warmup
- Statistics monitoring
- Cache management operations
- Production caching patterns

**8 Examples:**
1. Basic cache usage
2. Cache performance impact
3. LRU eviction
4. TTL expiration
5. Cache warmup
6. Cache statistics
7. Cache management
8. Production patterns

**Run it:**
```bash
python examples/caching_example.py     # All examples
python examples/caching_example.py 2   # Performance impact
python examples/caching_example.py 8   # Production patterns
```

**Key Features:**
- LRU eviction with OrderedDict
- TTL-based expiration
- Thread-safe operations
- Statistics tracking
- Warmup for common queries
- Hit rate optimization

### 4. Command-Line Interface

#### [CLI_EXAMPLES.md](CLI_EXAMPLES.md)
**What it shows:** Complete CLI usage guide
- `generate` - Generate synthetic data
- `train` - Train routing models
- `evaluate` - Evaluate performance
- `predict` - Make predictions
- `serve` - Start REST server
- Complete workflows
- Troubleshooting guide

**Read it:**
```bash
cat examples/CLI_EXAMPLES.md
# or open in your editor
```

**Covered Commands:**
```bash
# Generate data
funcroute generate --tools tools.json --output data.jsonl

# Train model
funcroute train --data train.jsonl --output model/ --tools tools.json

# Evaluate
funcroute evaluate --model model/ --data test.jsonl --visualize

# Predict
funcroute predict --model model/ --query "test query"

# Serve
funcroute serve --model model/ --port 8000 --cache-size 1000
```

## Running Examples

### Prerequisites

1. **Install FuncRoute:**
```bash
pip install funcroute
# or for development:
pip install -e .
```

2. **Install server dependencies (optional):**
```bash
pip install 'funcroute[server]'
```

3. **Train example model:**
```bash
cd examples
python batch_prediction_example.py  # Creates batch_example_model/
```

### Run Individual Examples

```bash
# Basic workflow
python examples/basic_example.py

# Batch predictions
python examples/batch_prediction_example.py

# Streaming predictions
python examples/streaming_prediction_example.py

# Async predictions
python examples/async_prediction_example.py

# Caching
python examples/caching_example.py

# Server (starts server, Ctrl+C to stop)
python examples/server_example.py
```

### Run Specific Sub-Examples

Most example files support running individual examples:

```bash
# Run example #3 from batch predictions
python examples/batch_prediction_example.py 3

# Run example #5 from async predictions
python examples/async_prediction_example.py 5

# Run example #2 from caching
python examples/caching_example.py 2
```

## Example Categories

### By Use Case

**High-Traffic APIs:**
- [server_example.py](server_example.py) - REST API deployment
- [caching_example.py](caching_example.py) - Cache optimization
- [async_prediction_example.py](async_prediction_example.py) - Async patterns

**Batch Processing:**
- [batch_prediction_example.py](batch_prediction_example.py) - Parallel batch
- [streaming_prediction_example.py](streaming_prediction_example.py) - Streaming patterns

**Real-Time Systems:**
- [streaming_prediction_example.py](streaming_prediction_example.py) - Low latency
- [async_prediction_example.py](async_prediction_example.py) - Concurrent requests

**Data Pipeline:**
- [basic_example.py](basic_example.py) - End-to-end workflow
- [streaming_prediction_example.py](streaming_prediction_example.py) - Pipeline stages

### By Feature

**Predictor:**
- All prediction examples demonstrate `Predictor` class
- `predict_batch()`, `predict_stream()`, `predict_async()`
- Callbacks, filtering, conditional termination

**Cache:**
- [caching_example.py](caching_example.py) - RouteCache and WarmupCache
- LRU eviction, TTL, statistics

**Server:**
- [server_example.py](server_example.py) - FastAPI integration
- Endpoints, middleware, custom apps

**CLI:**
- [CLI_EXAMPLES.md](CLI_EXAMPLES.md) - All CLI commands
- Complete workflows

## Performance Tips

### From Examples

1. **Use Caching** (from [caching_example.py](caching_example_example.py:example_2))
   - Can achieve 5-10x speedup for repeated queries
   - Aim for >80% hit rate in production

2. **Parallelize Batch Predictions** (from [batch_prediction_example.py](batch_prediction_example.py:example_2))
   - 4 workers: ~3x speedup
   - 8 workers: ~5x speedup
   - Adjust based on CPU cores

3. **Stream for Large Batches** (from [streaming_prediction_example.py](streaming_prediction_example.py:example_3))
   - Get first result ~10x faster than batch
   - Start processing before all complete

4. **Use Async for Web Apps** (from [async_prediction_example.py](async_prediction_example.py:example_5))
   - Better concurrency than threads
   - Perfect for FastAPI/aiohttp

5. **Warm Up Cache** (from [caching_example.py](caching_example.py:example_5))
   - Pre-populate common queries
   - Faster initial responses

## Common Patterns

### Pattern 1: High-Throughput API

```python
# From server_example.py and caching_example.py
from funcroute import FuncRoute
from funcroute.inference import Predictor, RouteCache
from funcroute.inference.server import run_server

router = FuncRoute.load("./model")
cache = RouteCache(max_size=10000, ttl_seconds=300)

run_server(router, host="0.0.0.0", port=8000, cache_size=10000)
```

### Pattern 2: Batch Processing Job

```python
# From batch_prediction_example.py
from funcroute import FuncRoute
from funcroute.inference import Predictor

router = FuncRoute.load("./model")
predictor = Predictor(router)

# Process large batch efficiently
results = predictor.predict_batch(
    queries,
    max_workers=8,
    show_progress=True
)
```

### Pattern 3: Real-Time Pipeline

```python
# From streaming_prediction_example.py
from funcroute import FuncRoute
from funcroute.inference import Predictor

router = FuncRoute.load("./model")
predictor = Predictor(router)

# Stream and process in real-time
for result in predictor.predict_stream(large_query_list):
    # Process immediately (e.g., save to DB)
    save_to_database(result)
```

### Pattern 4: Async Web Service

```python
# From async_prediction_example.py
from funcroute import FuncRoute
from funcroute.inference import Predictor, RouteCache

router = FuncRoute.load("./model")
cache = RouteCache(max_size=1000)
predictor = Predictor(router, cache=cache)

async def handle_request(query: str):
    result = await predictor.predict_async(query)
    return {"tool": result.tool, "confidence": result.confidence}
```

## Example Output

### Batch Prediction
```
================================================================================
Example 1: Basic Batch Prediction
================================================================================

Making 10 batch predictions...
100%|████████████████████████████████████████| 10/10 [00:00<00:00, 25.3 queries/s]

Results:
  Query 0: Find product type 0 → search_products (95.2%)
  Query 1: Find product type 1 → search_products (94.8%)
  ...

✅ Batch prediction completed in 0.39s
```

### Streaming Prediction
```
================================================================================
Example 1: Basic Streaming Prediction
================================================================================

Streaming 8 predictions...

Completed    Query                               Tool                 Conf
--------------------------------------------------------------------------------
1/8          Where is my order?                  manage_order         98.5%
2/8          Find wireless keyboards             search_products      97.2%
3/8          Return defective product            process_return       96.8%
...

✅ All 8 predictions completed in 0.45s
💡 Notice: Results may arrive out of order!
```

### Caching
```
================================================================================
Example 2: Cache Performance Impact
================================================================================

1️⃣  Without cache:
   Time: 2.15s
   Queries/sec: 46.5

2️⃣  With cache:
   Time: 0.38s
   Queries/sec: 263.2
   Cache hits: 95
   Cache hit rate: 95.0%

⚡ Speedup: 5.7x faster with cache!
💡 Cache reduced 95 redundant model calls
```

### Async Prediction
```
================================================================================
Example 2: Concurrent Async Predictions
================================================================================

🔄 Running 5 predictions concurrently...

Query                          Tool                 Confidence
--------------------------------------------------------------------------------
Where is my order?             manage_order         98.5%
Find laptops                   search_products      97.2%
Return this item               process_return       96.8%
Track package                  manage_order         99.1%
Search for monitors            search_products      98.3%

✅ Completed 5 predictions in 0.08s
💡 Concurrent execution allows parallel processing
```

## Troubleshooting

### Model Not Found Error

```
❌ Model not found. Please run batch_prediction_example.py first.
```

**Solution:** Train the example model first:
```bash
python examples/batch_prediction_example.py
```

### Import Errors

```
ModuleNotFoundError: No module named 'fastapi'
```

**Solution:** Install server dependencies:
```bash
pip install 'funcroute[server]'
```

### Async Test Warnings

```
PytestUnraisableExceptionWarning: async def functions are not natively supported
```

**Solution:** Install pytest-asyncio:
```bash
pip install pytest-asyncio
```

## Next Steps

1. **Start with basic_example.py** to understand the complete workflow
2. **Try batch_prediction_example.py** for production batch processing
3. **Explore caching_example.py** to optimize performance
4. **Read CLI_EXAMPLES.md** for command-line usage
5. **Check server_example.py** for API deployment

## Contributing

To add new examples:
1. Follow the existing pattern (example_N_* functions)
2. Include docstrings explaining what each example shows
3. Add to this README
4. Test with the shared `batch_example_model`

## Questions?

- Check [CLI_EXAMPLES.md](CLI_EXAMPLES.md) for CLI usage
- See individual example files for detailed comments
- Refer to main documentation in `../docs/`
