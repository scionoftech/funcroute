"""
Evaluator for measuring routing performance
"""

from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np

from funcroute.core.config import RouteResult
from funcroute.evaluation import metrics


class Evaluator:
    """
    Evaluate routing performance with comprehensive metrics.

    Example:
        >>> from funcroute import FuncRoute, Evaluator
        >>> router = FuncRoute.load("./model")
        >>> evaluator = Evaluator(router)
        >>> results = evaluator.evaluate(test_data)
        >>> evaluator.print_report(results)
    """

    def __init__(self, router=None):
        """
        Initialize evaluator.

        Args:
            router: FuncRoute instance (optional, can be passed to evaluate())
        """
        self.router = router

    def evaluate(
        self,
        test_data: List[Dict[str, str]],
        router=None,
        verbose: bool = True,
    ) -> Dict[str, any]:
        """
        Evaluate router on test data.

        Args:
            test_data: List of {"query": str, "tool": str} samples
            router: FuncRoute instance (uses self.router if not provided)
            verbose: Show progress bar

        Returns:
            Dict with comprehensive evaluation results

        Example:
            >>> test_data = [
            ...     {"query": "Where is my order?", "tool": "manage_order"},
            ...     {"query": "Find laptops", "tool": "search_products"},
            ... ]
            >>> results = evaluator.evaluate(test_data)
        """
        if router is None:
            router = self.router

        if router is None:
            raise ValueError("No router provided. Pass router to __init__ or evaluate()")

        if len(test_data) == 0:
            raise ValueError("Test data is empty")

        # Run predictions
        predictions = []
        ground_truth = []
        confidences = []
        latencies = []
        all_alternatives = []

        iterator = tqdm(test_data, desc="Evaluating") if verbose else test_data

        for sample in iterator:
            query = sample["query"]
            true_tool = sample["tool"]

            # Route
            result = router.route(query)

            predictions.append(result.tool)
            ground_truth.append(true_tool)
            confidences.append(result.confidence)
            latencies.append(result.latency_ms)

            # Build alternatives list including top prediction
            alts = [(result.tool, result.confidence)]
            if result.alternatives:
                alts.extend(result.alternatives)
            all_alternatives.append(alts)

        # Calculate metrics
        acc = metrics.accuracy(predictions, ground_truth)
        prec, rec, f1 = metrics.precision_recall_f1(predictions, ground_truth, average="macro")
        prec_micro, rec_micro, f1_micro = metrics.precision_recall_f1(
            predictions, ground_truth, average="micro"
        )
        prec_weighted, rec_weighted, f1_weighted = metrics.precision_recall_f1(
            predictions, ground_truth, average="weighted"
        )

        # Per-tool metrics
        tool_metrics = metrics.per_tool_metrics(predictions, ground_truth)

        # Confusion matrix
        conf_matrix, labels = metrics.confusion_matrix(predictions, ground_truth)

        # Calibration
        calibration = metrics.confidence_calibration(
            confidences, predictions, ground_truth, num_bins=10
        )

        # Latency
        latency = metrics.latency_stats(latencies)

        # Top-k accuracy
        top3_acc = metrics.top_k_accuracy(all_alternatives, ground_truth, k=3)
        top5_acc = metrics.top_k_accuracy(all_alternatives, ground_truth, k=5)

        return {
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1,
            "precision_micro": prec_micro,
            "recall_micro": rec_micro,
            "f1_micro": f1_micro,
            "precision_weighted": prec_weighted,
            "recall_weighted": rec_weighted,
            "f1_weighted": f1_weighted,
            "top3_accuracy": top3_acc,
            "top5_accuracy": top5_acc,
            "per_tool_metrics": tool_metrics,
            "confusion_matrix": conf_matrix,
            "confusion_matrix_labels": labels,
            "calibration": calibration,
            "latency": latency,
            "num_samples": len(test_data),
            "predictions": predictions,
            "ground_truth": ground_truth,
            "confidences": confidences,
        }

    def print_report(self, results: Dict[str, any]):
        """
        Print comprehensive evaluation report.

        Args:
            results: Results dict from evaluate()
        """
        print("\n" + "=" * 80)
        print("ROUTING EVALUATION REPORT")
        print("=" * 80)

        # Overall metrics
        print(f"\n📊 Overall Performance ({results['num_samples']} samples):")
        print(f"  Accuracy:           {results['accuracy']:.2%}")
        print(f"  Top-3 Accuracy:     {results['top3_accuracy']:.2%}")
        print(f"  Top-5 Accuracy:     {results['top5_accuracy']:.2%}")

        print(f"\n  Macro Averages:")
        print(f"    Precision:        {results['precision_macro']:.2%}")
        print(f"    Recall:           {results['recall_macro']:.2%}")
        print(f"    F1 Score:         {results['f1_macro']:.2%}")

        print(f"\n  Weighted Averages:")
        print(f"    Precision:        {results['precision_weighted']:.2%}")
        print(f"    Recall:           {results['recall_weighted']:.2%}")
        print(f"    F1 Score:         {results['f1_weighted']:.2%}")

        # Per-tool performance
        print(f"\n🎯 Per-Tool Performance:")
        print(f"  {'Tool':<30s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'Acc':>8s} {'Support':>8s}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for tool, tool_stats in sorted(results["per_tool_metrics"].items()):
            print(
                f"  {tool:<30s} "
                f"{tool_stats['precision']:>7.1%} "
                f"{tool_stats['recall']:>7.1%} "
                f"{tool_stats['f1']:>7.1%} "
                f"{tool_stats['accuracy']:>7.1%} "
                f"{tool_stats['support']:>8d}"
            )

        # Latency
        print(f"\n⚡ Latency Statistics (ms):")
        lat = results["latency"]
        print(f"  Mean:    {lat['mean']:>8.2f}")
        print(f"  Median:  {lat['median']:>8.2f}")
        print(f"  Std Dev: {lat['std']:>8.2f}")
        print(f"  Min:     {lat['min']:>8.2f}")
        print(f"  Max:     {lat['max']:>8.2f}")
        print(f"  P90:     {lat['p90']:>8.2f}")
        print(f"  P95:     {lat['p95']:>8.2f}")
        print(f"  P99:     {lat['p99']:>8.2f}")

        # Calibration
        print(f"\n📐 Confidence Calibration:")
        cal = results["calibration"]
        print(f"  Expected Calibration Error (ECE): {cal['ece']:.4f}")
        print(f"  (Lower is better, 0 = perfect calibration)")

        # Confusion matrix summary
        print(f"\n🔢 Confusion Matrix:")
        conf_matrix = results["confusion_matrix"]
        labels = results["confusion_matrix_labels"]

        # Print header
        header = "  True \\ Pred  "
        for label in labels:
            header += f"{label[:8]:>10s}"
        print(header)

        # Print rows
        for i, true_label in enumerate(labels):
            row = f"  {true_label:<13s}"
            for j in range(len(labels)):
                row += f"{conf_matrix[i][j]:>10d}"
            print(row)

        print("\n" + "=" * 80)

        # Overall verdict
        if results["accuracy"] >= 0.95:
            print("✅ EXCELLENT - Router is performing very well!")
        elif results["accuracy"] >= 0.85:
            print("✅ GOOD - Router is performing well")
        elif results["accuracy"] >= 0.70:
            print("⚠️  FAIR - Router could use improvement")
        else:
            print("❌ POOR - Router needs significant improvement")

        print("=" * 80)

    def cross_validate(
        self,
        data: List[Dict[str, str]],
        train_fn,
        n_folds: int = 5,
        verbose: bool = True,
    ) -> Dict[str, any]:
        """
        Perform k-fold cross-validation.

        Args:
            data: Full dataset
            train_fn: Function(train_data) -> router that trains and returns a router
            n_folds: Number of folds
            verbose: Show progress

        Returns:
            Dict with cross-validation results

        Example:
            >>> def train_func(train_data):
            ...     router = FuncRoute()
            ...     router.train(train_data, config=TrainingConfig(...))
            ...     return router
            >>> cv_results = evaluator.cross_validate(data, train_func, n_folds=5)
        """
        from funcroute.data.splitter import PatternGroupSplitter

        if len(data) < n_folds:
            raise ValueError(f"Not enough data for {n_folds} folds")

        # Split into folds using pattern grouping
        splitter = PatternGroupSplitter()

        # For k-fold, we'll do manual fold creation
        fold_size = len(data) // n_folds
        fold_results = []

        for fold_idx in range(n_folds):
            if verbose:
                print(f"\n{'='*80}")
                print(f"Fold {fold_idx + 1}/{n_folds}")
                print(f"{'='*80}")

            # Create test fold
            test_start = fold_idx * fold_size
            test_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else len(data)
            test_fold = data[test_start:test_end]
            train_fold = data[:test_start] + data[test_end:]

            if verbose:
                print(f"Train size: {len(train_fold)}, Test size: {len(test_fold)}")

            # Train router
            router = train_fn(train_fold)

            # Evaluate
            results = self.evaluate(test_fold, router=router, verbose=verbose)
            fold_results.append(results)

            if verbose:
                print(f"\nFold {fold_idx + 1} Accuracy: {results['accuracy']:.2%}")

        # Aggregate results
        cv_summary = {
            "n_folds": n_folds,
            "fold_results": fold_results,
            "accuracy_mean": np.mean([r["accuracy"] for r in fold_results]),
            "accuracy_std": np.std([r["accuracy"] for r in fold_results]),
            "f1_macro_mean": np.mean([r["f1_macro"] for r in fold_results]),
            "f1_macro_std": np.std([r["f1_macro"] for r in fold_results]),
            "latency_mean": np.mean([r["latency"]["mean"] for r in fold_results]),
            "latency_std": np.std([r["latency"]["mean"] for r in fold_results]),
        }

        if verbose:
            print(f"\n{'='*80}")
            print(f"CROSS-VALIDATION SUMMARY ({n_folds} folds)")
            print(f"{'='*80}")
            print(f"  Accuracy:     {cv_summary['accuracy_mean']:.2%} ± {cv_summary['accuracy_std']:.2%}")
            print(f"  F1 (macro):   {cv_summary['f1_macro_mean']:.2%} ± {cv_summary['f1_macro_std']:.2%}")
            print(f"  Latency (ms): {cv_summary['latency_mean']:.2f} ± {cv_summary['latency_std']:.2f}")
            print(f"{'='*80}")

        return cv_summary

    def compare_routers(
        self,
        routers: Dict[str, any],
        test_data: List[Dict[str, str]],
        verbose: bool = True,
    ) -> Dict[str, Dict[str, any]]:
        """
        Compare multiple routers on same test data.

        Args:
            routers: Dict mapping router name to router instance
            test_data: Test data
            verbose: Show progress

        Returns:
            Dict mapping router name to evaluation results

        Example:
            >>> routers = {
            ...     "baseline": baseline_router,
            ...     "improved": improved_router,
            ... }
            >>> comparison = evaluator.compare_routers(routers, test_data)
        """
        results = {}

        for name, router in routers.items():
            if verbose:
                print(f"\n{'='*80}")
                print(f"Evaluating: {name}")
                print(f"{'='*80}")

            results[name] = self.evaluate(test_data, router=router, verbose=verbose)

            if verbose:
                print(f"\n{name} - Accuracy: {results[name]['accuracy']:.2%}")

        # Print comparison
        if verbose:
            print(f"\n{'='*80}")
            print("ROUTER COMPARISON")
            print(f"{'='*80}")
            print(
                f"  {'Router':<20s} {'Accuracy':>10s} {'F1':>10s} {'Latency':>12s} {'Top-3':>10s}"
            )
            print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

            for name, result in results.items():
                print(
                    f"  {name:<20s} "
                    f"{result['accuracy']:>9.2%} "
                    f"{result['f1_macro']:>9.2%} "
                    f"{result['latency']['mean']:>10.2f}ms "
                    f"{result['top3_accuracy']:>9.2%}"
                )
            print(f"{'='*80}")

        return results
