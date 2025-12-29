"""
REST API Server Example

Demonstrates how to serve a FuncRoute model as a REST API using FastAPI.

NOTE: Requires FastAPI and uvicorn:
    pip install 'funcroute[server]'
    or
    pip install fastapi uvicorn
"""

from funcroute import FuncRoute, TrainingConfig
from funcroute.core.config import ToolDefinition
from funcroute.data.generator import SyntheticDataGenerator
from funcroute.data.splitter import PatternGroupSplitter
from funcroute.inference.server import create_app, run_server


def train_example_model():
    """Train a simple model for the example"""
    print("Training example model...")

    # Define tools
    tools = [
        ToolDefinition(
            name="manage_order",
            signature="manage_order(order_id: str) -> dict",
            description="Track and manage customer orders",
            examples=["Where is my order?", "Track package"],
        ),
        ToolDefinition(
            name="search_products",
            signature="search_products(query: str) -> list",
            description="Search for products",
            examples=["Find laptops", "Show me shoes"],
        ),
        ToolDefinition(
            name="process_return",
            signature="process_return(order_id: str) -> dict",
            description="Process returns and refunds",
            examples=["I want to return this", "Get a refund"],
        ),
    ]

    # Generate synthetic data
    generator = SyntheticDataGenerator(method="rule_based")
    data = generator.generate(tools, num_variations=30, num_samples=300)

    # Split data
    splitter = PatternGroupSplitter()
    train_data, val_data, _ = splitter.split(data)

    # Train
    router = FuncRoute()
    router.train(
        train_data=train_data,
        val_data=val_data,
        tools=tools,
        config=TrainingConfig(
            output_dir="./server_example_model",
            num_epochs=1,  # Use more in production
            batch_size=4,
        ),
    )

    print("✅ Model trained!")
    return router


def example_1_basic_server():
    """
    Example 1: Basic server without cache

    Start server with:
        python examples/server_example.py
    """
    print("=" * 80)
    print("Example 1: Basic Server (No Cache)")
    print("=" * 80)

    # Train or load model
    router = train_example_model()
    # Or load existing:  router = FuncRoute.load("./server_example_model")

    # Run server
    run_server(
        router,
        host="0.0.0.0",
        port=8000,
        cache_size=None,  # No cache
    )


def example_2_server_with_cache():
    """
    Example 2: Server with caching enabled

    Start server with:
        python -c "from examples.server_example import example_2_server_with_cache; example_2_server_with_cache()"
    """
    print("=" * 80)
    print("Example 2: Server with Cache")
    print("=" * 80)

    # Load model
    router = FuncRoute.load("./server_example_model")

    # Run server with cache
    run_server(
        router,
        host="0.0.0.0",
        port=8000,
        cache_size=1000,  # LRU cache with 1000 entries
        cache_ttl=300,  # 5 minute TTL
    )


def example_3_custom_app():
    """
    Example 3: Create custom app with additional endpoints

    This shows how to extend the FuncRoute server with custom endpoints.
    """
    print("=" * 80)
    print("Example 3: Custom App with Additional Endpoints")
    print("=" * 80)

    from funcroute.inference import RouteCache, Predictor

    # Load model
    router = FuncRoute.load("./server_example_model")

    # Create cache and predictor
    cache = RouteCache(max_size=500, ttl_seconds=600)
    predictor = Predictor(router, cache=cache)

    # Create base app
    app = create_app(router, predictor=predictor, cache=cache)

    # Add custom endpoint
    @app.get("/custom/ping")
    async def custom_ping():
        """Custom endpoint"""
        return {"message": "pong", "service": "funcroute"}

    @app.get("/custom/model-info")
    async def model_info():
        """Get model information"""
        return {
            "model_path": "./server_example_model",
            "cache_enabled": True,
            "cache_size": 500,
            "cache_ttl": 600,
        }

    # Run with uvicorn
    import uvicorn

    print("\n🚀 Starting custom FuncRoute server on http://0.0.0.0:8000")
    print("   Custom endpoints:")
    print("     - GET /custom/ping")
    print("     - GET /custom/model-info")

    uvicorn.run(app, host="0.0.0.0", port=8000)


def example_4_test_api():
    """
    Example 4: Test the API using requests

    NOTE: Server must be running first (run example_1 or example_2)
    """
    print("=" * 80)
    print("Example 4: Testing the API")
    print("=" * 80)

    try:
        import requests
    except ImportError:
        print("❌ requests library not installed. Install with: pip install requests")
        return

    base_url = "http://localhost:8000"

    # Test health endpoint
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")

    # Test single route
    print("\n2. Testing single route...")
    payload = {"query": "Where is my order?", "include_alternatives": True}
    response = requests.post(f"{base_url}/route", json=payload)
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Query: {result['query']}")
    print(f"   Tool: {result['tool']}")
    print(f"   Confidence: {result['confidence']:.2%}")
    print(f"   Latency: {result['latency_ms']:.1f}ms")

    # Test batch route
    print("\n3. Testing batch route...")
    payload = {
        "queries": [
            "Where is my order?",
            "Find laptops under $1000",
            "I want to return this item",
        ],
        "include_alternatives": False,
    }
    response = requests.post(f"{base_url}/route/batch", json=payload)
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Total queries: {result['total_queries']}")
    print(f"   Total latency: {result['total_latency_ms']:.1f}ms")
    for r in result["results"]:
        print(f"     - '{r['query']}' → {r['tool']} ({r['confidence']:.2%})")

    # Test cache stats (if cache enabled)
    print("\n4. Testing cache stats...")
    try:
        response = requests.get(f"{base_url}/cache/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"   Hit rate: {stats['hit_rate']:.2%}")
            print(f"   Size: {stats['size']}/{stats['max_size']}")
            print(f"   Hits: {stats['hits']}, Misses: {stats['misses']}")
        else:
            print(f"   Cache not enabled or error: {response.status_code}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test server stats
    print("\n5. Testing server stats...")
    response = requests.get(f"{base_url}/stats")
    print(f"   Status: {response.status_code}")
    stats = response.json()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Total batch requests: {stats['total_batch_requests']}")
    print(f"   Uptime: {stats['uptime_seconds']:.1f}s")

    print("\n✅ API tests complete!")


def example_5_curl_commands():
    """
    Example 5: Example curl commands for API testing

    Copy and paste these commands to test the API from command line.
    """
    print("=" * 80)
    print("Example 5: Curl Command Examples")
    print("=" * 80)

    curl_commands = """
# Health check
curl http://localhost:8000/health

# Single route
curl -X POST http://localhost:8000/route \\
  -H "Content-Type: application/json" \\
  -d '{"query": "Where is my order?", "include_alternatives": true}'

# Batch route
curl -X POST http://localhost:8000/route/batch \\
  -H "Content-Type: application/json" \\
  -d '{
    "queries": [
      "Where is my order?",
      "Find laptops",
      "Return this item"
    ],
    "include_alternatives": false
  }'

# Cache stats
curl http://localhost:8000/cache/stats

# Clear cache
curl -X POST http://localhost:8000/cache/clear

# Server stats
curl http://localhost:8000/stats

# OpenAPI docs
# Visit: http://localhost:8000/docs
"""

    print(curl_commands)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example = sys.argv[1]
        if example == "1":
            example_1_basic_server()
        elif example == "2":
            example_2_server_with_cache()
        elif example == "3":
            example_3_custom_app()
        elif example == "4":
            example_4_test_api()
        elif example == "5":
            example_5_curl_commands()
        else:
            print("Usage: python server_example.py [1|2|3|4|5]")
            print("  1: Basic server (no cache)")
            print("  2: Server with cache")
            print("  3: Custom app with additional endpoints")
            print("  4: Test the API (server must be running)")
            print("  5: Show curl command examples")
    else:
        # Default: run basic server
        example_1_basic_server()
