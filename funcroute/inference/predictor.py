"""
Predictor for efficient batch and streaming inference
"""

from typing import List, Dict, Iterator, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from tqdm import tqdm

from funcroute.core.config import RouteResult


class Predictor:
    """
    Efficient prediction with batch processing, streaming, and async support.

    Example:
        >>> from funcroute import FuncRoute
        >>> from funcroute.inference import Predictor
        >>>
        >>> router = FuncRoute.load("./model")
        >>> predictor = Predictor(router)
        >>>
        >>> # Batch prediction
        >>> queries = ["Where is my order?", "Find laptops"]
        >>> results = predictor.predict_batch(queries)
        >>>
        >>> # Streaming prediction
        >>> for result in predictor.predict_stream(queries):
        ...     print(f"{result.query} -> {result.tool}")
    """

    def __init__(self, router, cache=None):
        """
        Initialize predictor.

        Args:
            router: FuncRoute instance
            cache: Optional RouteCache instance for caching
        """
        self.router = router
        self.cache = cache

    def predict_batch(
        self,
        queries: List[str],
        max_workers: Optional[int] = None,
        show_progress: bool = True,
        use_cache: bool = True,
    ) -> List[RouteResult]:
        """
        Predict multiple queries in batch with parallel processing.

        Args:
            queries: List of query strings
            max_workers: Max parallel workers (default: min(32, cpu_count + 4))
            show_progress: Show progress bar
            use_cache: Use cache if available

        Returns:
            List of RouteResult objects

        Example:
            >>> queries = ["query1", "query2", "query3"]
            >>> results = predictor.predict_batch(queries, max_workers=4)
        """
        if len(queries) == 0:
            return []

        # Single query - no parallelization needed
        if len(queries) == 1:
            result = self._predict_single(queries[0], use_cache=use_cache)
            return [result]

        results = [None] * len(queries)  # Preserve order

        # Create progress bar
        iterator = tqdm(total=len(queries), desc="Predicting") if show_progress else None

        # Parallel prediction
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(self._predict_single, query, use_cache): idx
                for idx, query in enumerate(queries)
            }

            # Collect results
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                except Exception as e:
                    # Create error result
                    results[idx] = RouteResult(
                        query=queries[idx],
                        tool="error",
                        confidence=0.0,
                        latency_ms=0.0,
                        metadata={"error": str(e)},
                    )

                if iterator:
                    iterator.update(1)

        if iterator:
            iterator.close()

        return results

    def predict_stream(
        self,
        queries: List[str],
        max_workers: Optional[int] = None,
        use_cache: bool = True,
    ) -> Iterator[RouteResult]:
        """
        Stream predictions as they complete (results may be out of order).

        Useful for processing large batches where you want to start
        handling results before all predictions complete.

        Args:
            queries: List of query strings
            max_workers: Max parallel workers
            use_cache: Use cache if available

        Yields:
            RouteResult objects as they complete

        Example:
            >>> for result in predictor.predict_stream(large_query_list):
            ...     process_result(result)
        """
        if len(queries) == 0:
            return

        # Single query
        if len(queries) == 1:
            yield self._predict_single(queries[0], use_cache=use_cache)
            return

        # Parallel streaming
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._predict_single, query, use_cache)
                for query in queries
            ]

            for future in as_completed(futures):
                try:
                    yield future.result()
                except Exception as e:
                    # Yield error result
                    yield RouteResult(
                        query="",
                        tool="error",
                        confidence=0.0,
                        latency_ms=0.0,
                        metadata={"error": str(e)},
                    )

    async def predict_async(
        self,
        query: str,
        use_cache: bool = True,
    ) -> RouteResult:
        """
        Async prediction for single query.

        Args:
            query: Query string
            use_cache: Use cache if available

        Returns:
            RouteResult

        Example:
            >>> result = await predictor.predict_async("Where is my order?")
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._predict_single, query, use_cache
        )

    async def predict_batch_async(
        self,
        queries: List[str],
        use_cache: bool = True,
    ) -> List[RouteResult]:
        """
        Async batch prediction.

        Args:
            queries: List of query strings
            use_cache: Use cache if available

        Returns:
            List of RouteResult objects

        Example:
            >>> results = await predictor.predict_batch_async(queries)
        """
        if len(queries) == 0:
            return []

        # Create tasks
        tasks = [self.predict_async(query, use_cache) for query in queries]

        # Wait for all
        return await asyncio.gather(*tasks)

    def predict_with_callback(
        self,
        queries: List[str],
        callback: Callable[[RouteResult], None],
        max_workers: Optional[int] = None,
        use_cache: bool = True,
    ):
        """
        Predict with callback for each result (streaming pattern).

        Useful for real-time processing or logging.

        Args:
            queries: List of query strings
            callback: Function to call with each result
            max_workers: Max parallel workers
            use_cache: Use cache if available

        Example:
            >>> def process(result):
            ...     print(f"Processed: {result.query} -> {result.tool}")
            >>>
            >>> predictor.predict_with_callback(queries, process)
        """
        for result in self.predict_stream(queries, max_workers, use_cache):
            callback(result)

    def predict_with_filter(
        self,
        queries: List[str],
        filter_fn: Callable[[RouteResult], bool],
        max_workers: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[RouteResult]:
        """
        Predict and filter results.

        Args:
            queries: List of query strings
            filter_fn: Function that returns True to keep result
            max_workers: Max parallel workers
            use_cache: Use cache if available

        Returns:
            Filtered list of RouteResult objects

        Example:
            >>> # Only keep high-confidence results
            >>> high_conf = predictor.predict_with_filter(
            ...     queries,
            ...     lambda r: r.confidence > 0.8
            ... )
        """
        results = self.predict_batch(
            queries, max_workers=max_workers, show_progress=False, use_cache=use_cache
        )
        return [r for r in results if filter_fn(r)]

    def predict_until(
        self,
        queries: List[str],
        condition: Callable[[List[RouteResult]], bool],
        max_workers: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[RouteResult]:
        """
        Stream predictions until condition is met.

        Args:
            queries: List of query strings
            condition: Function that takes accumulated results and returns True to stop
            max_workers: Max parallel workers
            use_cache: Use cache if available

        Returns:
            List of RouteResult objects up to condition

        Example:
            >>> # Stop after 5 high-confidence results
            >>> results = predictor.predict_until(
            ...     queries,
            ...     lambda results: len([r for r in results if r.confidence > 0.9]) >= 5
            ... )
        """
        results = []

        for result in self.predict_stream(queries, max_workers, use_cache):
            results.append(result)
            if condition(results):
                break

        return results

    def _predict_single(self, query: str, use_cache: bool = True) -> RouteResult:
        """
        Predict single query with optional caching.

        Args:
            query: Query string
            use_cache: Use cache if available

        Returns:
            RouteResult
        """
        # Check cache
        if use_cache and self.cache is not None:
            cached = self.cache.get(query)
            if cached is not None:
                return cached

        # Predict
        result = self.router.route(query)

        # Store in cache
        if use_cache and self.cache is not None:
            self.cache.put(query, result)

        return result

    def get_cache_stats(self) -> Optional[Dict[str, any]]:
        """
        Get cache statistics if cache is enabled.

        Returns:
            Dict with cache stats or None

        Example:
            >>> stats = predictor.get_cache_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
        """
        if self.cache is None:
            return None

        return self.cache.get_stats()

    def clear_cache(self):
        """Clear the cache if enabled."""
        if self.cache is not None:
            self.cache.clear()
