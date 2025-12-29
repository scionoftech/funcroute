"""
FastAPI REST server for FuncRoute
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import time

# FastAPI imports - optional dependency
try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class RouteRequest(BaseModel):
    """Single route request"""
    query: str = Field(..., description="Query string to route", min_length=1)
    include_alternatives: bool = Field(
        default=False, description="Include alternative predictions"
    )
    use_cache: bool = Field(default=True, description="Use cache if available")


class BatchRouteRequest(BaseModel):
    """Batch route request"""
    queries: List[str] = Field(..., description="List of queries to route", min_items=1)
    include_alternatives: bool = Field(
        default=False, description="Include alternative predictions"
    )
    use_cache: bool = Field(default=True, description="Use cache if available")


class RouteResponse(BaseModel):
    """Route response"""
    query: str
    tool: str
    confidence: float
    latency_ms: float
    alternatives: Optional[List[Dict[str, float]]] = None
    metadata: Optional[Dict] = None


class BatchRouteResponse(BaseModel):
    """Batch route response"""
    results: List[RouteResponse]
    total_queries: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    cache_enabled: bool
    cache_stats: Optional[Dict] = None
    uptime_seconds: float


class StatsResponse(BaseModel):
    """Server statistics response"""
    total_requests: int
    total_batch_requests: int
    cache_stats: Optional[Dict] = None
    uptime_seconds: float


def create_app(router, predictor=None, cache=None, enable_cors: bool = True) -> FastAPI:
    """
    Create FastAPI application for FuncRoute.

    Args:
        router: FuncRoute instance
        predictor: Optional Predictor instance (created if not provided)
        cache: Optional RouteCache instance
        enable_cors: Enable CORS middleware

    Returns:
        FastAPI application

    Example:
        >>> from funcroute import FuncRoute
        >>> from funcroute.inference import create_app
        >>>
        >>> router = FuncRoute.load("./model")
        >>> app = create_app(router)
        >>>
        >>> # Run with: uvicorn app:app --host 0.0.0.0 --port 8000
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is not installed. Install with: pip install 'funcroute[server]' "
            "or pip install fastapi uvicorn"
        )

    from funcroute.inference import Predictor

    # Create predictor if not provided
    if predictor is None:
        predictor = Predictor(router, cache=cache)

    # Create app
    app = FastAPI(
        title="FuncRoute API",
        description="Intelligent function/tool routing API",
        version="0.1.0",
    )

    # Enable CORS if requested
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Server state
    start_time = time.time()
    stats = {
        "total_requests": 0,
        "total_batch_requests": 0,
    }

    # Routes
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint"""
        return {
            "name": "FuncRoute API",
            "version": "0.1.0",
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint"""
        cache_stats = predictor.get_cache_stats() if cache is not None else None

        return HealthResponse(
            status="healthy",
            model_loaded=True,
            cache_enabled=cache is not None,
            cache_stats=cache_stats,
            uptime_seconds=time.time() - start_time,
        )

    @app.get("/stats", response_model=StatsResponse)
    async def get_stats():
        """Get server statistics"""
        cache_stats = predictor.get_cache_stats() if cache is not None else None

        return StatsResponse(
            total_requests=stats["total_requests"],
            total_batch_requests=stats["total_batch_requests"],
            cache_stats=cache_stats,
            uptime_seconds=time.time() - start_time,
        )

    @app.post("/route", response_model=RouteResponse)
    async def route(request: RouteRequest):
        """
        Route a single query.

        Args:
            request: RouteRequest with query

        Returns:
            RouteResponse with routing result
        """
        stats["total_requests"] += 1

        try:
            # Predict
            result = predictor._predict_single(
                request.query, use_cache=request.use_cache
            )

            # Build response
            response = RouteResponse(
                query=result.query,
                tool=result.tool,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
                metadata=result.metadata,
            )

            # Include alternatives if requested
            if request.include_alternatives and result.alternatives:
                response.alternatives = [
                    {"tool": tool, "confidence": conf}
                    for tool, conf in result.alternatives
                ]

            return response

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Routing error: {str(e)}",
            )

    @app.post("/route/batch", response_model=BatchRouteResponse)
    async def route_batch(request: BatchRouteRequest):
        """
        Route multiple queries in batch.

        Args:
            request: BatchRouteRequest with queries

        Returns:
            BatchRouteResponse with all routing results
        """
        stats["total_batch_requests"] += 1
        stats["total_requests"] += len(request.queries)

        try:
            start = time.time()

            # Batch predict
            results = predictor.predict_batch(
                request.queries,
                show_progress=False,
                use_cache=request.use_cache,
            )

            total_latency = (time.time() - start) * 1000  # Convert to ms

            # Build responses
            route_responses = []
            for result in results:
                response = RouteResponse(
                    query=result.query,
                    tool=result.tool,
                    confidence=result.confidence,
                    latency_ms=result.latency_ms,
                    metadata=result.metadata,
                )

                # Include alternatives if requested
                if request.include_alternatives and result.alternatives:
                    response.alternatives = [
                        {"tool": tool, "confidence": conf}
                        for tool, conf in result.alternatives
                    ]

                route_responses.append(response)

            return BatchRouteResponse(
                results=route_responses,
                total_queries=len(request.queries),
                total_latency_ms=total_latency,
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch routing error: {str(e)}",
            )

    @app.post("/cache/clear")
    async def clear_cache():
        """Clear the routing cache"""
        if cache is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cache is not enabled",
            )

        predictor.clear_cache()
        return {"status": "success", "message": "Cache cleared"}

    @app.get("/cache/stats")
    async def cache_stats():
        """Get cache statistics"""
        if cache is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cache is not enabled",
            )

        stats_data = predictor.get_cache_stats()
        return stats_data

    return app


def run_server(
    router,
    host: str = "0.0.0.0",
    port: int = 8000,
    cache_size: Optional[int] = 1000,
    cache_ttl: Optional[int] = 300,
    enable_cors: bool = True,
    **uvicorn_kwargs,
):
    """
    Run FuncRoute server with uvicorn.

    Args:
        router: FuncRoute instance
        host: Host to bind to
        port: Port to bind to
        cache_size: Cache max size (None = no cache)
        cache_ttl: Cache TTL in seconds (None = no expiration)
        enable_cors: Enable CORS
        **uvicorn_kwargs: Additional uvicorn configuration

    Example:
        >>> from funcroute import FuncRoute
        >>> from funcroute.inference.server import run_server
        >>>
        >>> router = FuncRoute.load("./model")
        >>> run_server(router, host="0.0.0.0", port=8000)
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is not installed. Install with: pip install 'funcroute[server]' "
            "or pip install uvicorn"
        )

    from funcroute.inference import RouteCache, Predictor

    # Create cache if requested
    cache = None
    if cache_size is not None:
        cache = RouteCache(max_size=cache_size, ttl_seconds=cache_ttl)

    # Create predictor
    predictor = Predictor(router, cache=cache)

    # Create app
    app = create_app(router, predictor=predictor, cache=cache, enable_cors=enable_cors)

    # Run server
    print(f"\n🚀 Starting FuncRoute server on http://{host}:{port}")
    print(f"   API docs: http://{host}:{port}/docs")
    print(f"   Cache: {'Enabled' if cache else 'Disabled'}")
    if cache:
        print(f"   Cache size: {cache_size}, TTL: {cache_ttl}s")

    uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)
