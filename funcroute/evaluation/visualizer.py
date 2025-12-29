"""
Visualization tools for evaluation results
"""

from typing import Dict, List, Optional
import numpy as np


class Visualizer:
    """
    Create visualizations for routing evaluation results.

    Requires matplotlib and seaborn for plotting.

    Example:
        >>> from funcroute.evaluation import Visualizer
        >>> viz = Visualizer()
        >>> viz.plot_confusion_matrix(results, save_path="confusion.png")
    """

    def __init__(self):
        """Initialize visualizer"""
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if visualization dependencies are available"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            self.plt = plt
            self.sns = sns
            self.available = True
        except ImportError:
            self.available = False
            print(
                "⚠️  Warning: matplotlib and/or seaborn not installed. "
                "Install with: pip install matplotlib seaborn"
            )

    def plot_confusion_matrix(
        self,
        results: Dict[str, any],
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8),
        normalize: bool = False,
    ):
        """
        Plot confusion matrix heatmap.

        Args:
            results: Results dict from Evaluator.evaluate()
            save_path: Path to save figure (optional)
            figsize: Figure size (width, height)
            normalize: Normalize by row (show percentages)

        Example:
            >>> viz.plot_confusion_matrix(results, save_path="confusion.png")
        """
        if not self.available:
            print("Cannot plot: matplotlib/seaborn not available")
            return

        matrix = results["confusion_matrix"]
        labels = results["confusion_matrix_labels"]

        if normalize:
            # Normalize by row (true label)
            matrix = matrix.astype(float)
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            matrix = matrix / row_sums

        # Create figure
        fig, ax = self.plt.subplots(figsize=figsize)

        # Plot heatmap
        self.sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "Percentage" if normalize else "Count"},
        )

        ax.set_xlabel("Predicted Tool", fontsize=12)
        ax.set_ylabel("True Tool", fontsize=12)
        ax.set_title(
            "Confusion Matrix" + (" (Normalized)" if normalize else ""), fontsize=14
        )

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved confusion matrix to {save_path}")

        self.plt.show()

    def plot_confidence_distribution(
        self,
        results: Dict[str, any],
        save_path: Optional[str] = None,
        figsize: tuple = (12, 5),
    ):
        """
        Plot confidence score distribution for correct vs incorrect predictions.

        Args:
            results: Results dict from Evaluator.evaluate()
            save_path: Path to save figure (optional)
            figsize: Figure size

        Example:
            >>> viz.plot_confidence_distribution(results)
        """
        if not self.available:
            print("Cannot plot: matplotlib/seaborn not available")
            return

        predictions = results["predictions"]
        ground_truth = results["ground_truth"]
        confidences = results["confidences"]

        # Separate correct and incorrect
        correct_confs = [
            c for c, p, g in zip(confidences, predictions, ground_truth) if p == g
        ]
        incorrect_confs = [
            c for c, p, g in zip(confidences, predictions, ground_truth) if p != g
        ]

        # Create figure
        fig, (ax1, ax2) = self.plt.subplots(1, 2, figsize=figsize)

        # Plot histograms
        ax1.hist(correct_confs, bins=20, alpha=0.7, color="green", edgecolor="black")
        ax1.set_xlabel("Confidence Score")
        ax1.set_ylabel("Count")
        ax1.set_title(f"Correct Predictions (n={len(correct_confs)})")
        ax1.grid(alpha=0.3)

        ax2.hist(incorrect_confs, bins=20, alpha=0.7, color="red", edgecolor="black")
        ax2.set_xlabel("Confidence Score")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Incorrect Predictions (n={len(incorrect_confs)})")
        ax2.grid(alpha=0.3)

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved confidence distribution to {save_path}")

        self.plt.show()

    def plot_calibration_curve(
        self,
        results: Dict[str, any],
        save_path: Optional[str] = None,
        figsize: tuple = (8, 8),
    ):
        """
        Plot calibration curve (reliability diagram).

        Args:
            results: Results dict from Evaluator.evaluate()
            save_path: Path to save figure (optional)
            figsize: Figure size

        Example:
            >>> viz.plot_calibration_curve(results)
        """
        if not self.available:
            print("Cannot plot: matplotlib/seaborn not available")
            return

        calibration = results["calibration"]

        bin_confidences = calibration["bin_confidences"]
        bin_accuracies = calibration["bin_accuracies"]
        bin_counts = calibration["bin_counts"]

        # Create figure
        fig, ax = self.plt.subplots(figsize=figsize)

        # Plot calibration curve
        ax.plot(
            bin_confidences,
            bin_accuracies,
            marker="o",
            markersize=8,
            linestyle="-",
            linewidth=2,
            label="Model",
        )

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

        # Add bin counts as text
        for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
            ax.annotate(
                f"n={count}",
                (conf, acc),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                alpha=0.7,
            )

        ax.set_xlabel("Predicted Confidence", fontsize=12)
        ax.set_ylabel("Actual Accuracy", fontsize=12)
        ax.set_title(
            f"Calibration Curve (ECE: {calibration['ece']:.4f})", fontsize=14
        )
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved calibration curve to {save_path}")

        self.plt.show()

    def plot_per_tool_performance(
        self,
        results: Dict[str, any],
        metric: str = "f1",
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6),
    ):
        """
        Plot per-tool performance bar chart.

        Args:
            results: Results dict from Evaluator.evaluate()
            metric: Metric to plot ("f1", "precision", "recall", "accuracy")
            save_path: Path to save figure (optional)
            figsize: Figure size

        Example:
            >>> viz.plot_per_tool_performance(results, metric="f1")
        """
        if not self.available:
            print("Cannot plot: matplotlib/seaborn not available")
            return

        tool_metrics = results["per_tool_metrics"]

        tools = sorted(tool_metrics.keys())
        values = [tool_metrics[tool][metric] for tool in tools]
        supports = [tool_metrics[tool]["support"] for tool in tools]

        # Create figure
        fig, ax = self.plt.subplots(figsize=figsize)

        # Create bars
        bars = ax.bar(range(len(tools)), values, color="skyblue", edgecolor="black")

        # Color bars by performance
        for i, (bar, val) in enumerate(zip(bars, values)):
            if val >= 0.9:
                bar.set_color("green")
            elif val >= 0.7:
                bar.set_color("yellow")
            else:
                bar.set_color("red")

        # Add value labels on bars
        for i, (val, support) in enumerate(zip(values, supports)):
            ax.text(
                i,
                val + 0.02,
                f"{val:.1%}\n(n={support})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(range(len(tools)))
        ax.set_xticklabels(tools, rotation=45, ha="right")
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f"Per-Tool {metric.capitalize()} Score", fontsize=14)
        ax.set_ylim([0, 1.1])
        ax.grid(axis="y", alpha=0.3)

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved per-tool performance to {save_path}")

        self.plt.show()

    def plot_latency_distribution(
        self,
        results: Dict[str, any],
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6),
    ):
        """
        Plot latency distribution with percentiles.

        Args:
            results: Results dict from Evaluator.evaluate()
            save_path: Path to save figure (optional)
            figsize: Figure size

        Example:
            >>> viz.plot_latency_distribution(results)
        """
        if not self.available:
            print("Cannot plot: matplotlib/seaborn not available")
            return

        # Get latency data (we need raw latencies, not just stats)
        # For now, use the stats to create a representative visualization
        lat_stats = results["latency"]

        fig, ax = self.plt.subplots(figsize=figsize)

        # Create box plot representation
        data = {
            "Metric": ["Mean", "Median", "P90", "P95", "P99"],
            "Latency (ms)": [
                lat_stats["mean"],
                lat_stats["median"],
                lat_stats["p90"],
                lat_stats["p95"],
                lat_stats["p99"],
            ],
        }

        bars = ax.bar(data["Metric"], data["Latency (ms)"], color="lightcoral", edgecolor="black")

        # Add value labels
        for bar, val in zip(bars, data["Latency (ms)"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(data["Latency (ms)"]) * 0.02,
                f"{val:.2f}",
                ha="center",
                va="bottom",
            )

        ax.set_ylabel("Latency (ms)", fontsize=12)
        ax.set_title("Latency Statistics", fontsize=14)
        ax.grid(axis="y", alpha=0.3)

        self.plt.tight_layout()

        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved latency distribution to {save_path}")

        self.plt.show()

    def create_evaluation_dashboard(
        self,
        results: Dict[str, any],
        save_path: Optional[str] = None,
        figsize: tuple = (16, 12),
    ):
        """
        Create comprehensive dashboard with all key visualizations.

        Args:
            results: Results dict from Evaluator.evaluate()
            save_path: Path to save figure (optional)
            figsize: Figure size

        Example:
            >>> viz.create_evaluation_dashboard(results, save_path="dashboard.png")
        """
        if not self.available:
            print("Cannot plot: matplotlib/seaborn not available")
            return

        fig = self.plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0:2, 0])
        matrix = results["confusion_matrix"]
        labels = results["confusion_matrix_labels"]
        self.sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax1
        )
        ax1.set_title("Confusion Matrix")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("True")

        # 2. Per-tool F1
        ax2 = fig.add_subplot(gs[0, 1])
        tool_metrics = results["per_tool_metrics"]
        tools = sorted(tool_metrics.keys())
        f1_scores = [tool_metrics[tool]["f1"] for tool in tools]
        ax2.barh(tools, f1_scores, color="skyblue")
        ax2.set_xlabel("F1 Score")
        ax2.set_title("Per-Tool F1 Scores")
        ax2.set_xlim([0, 1])

        # 3. Calibration curve
        ax3 = fig.add_subplot(gs[1, 1])
        cal = results["calibration"]
        if cal["bin_confidences"]:
            ax3.plot(cal["bin_confidences"], cal["bin_accuracies"], "o-", label="Model")
            ax3.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
            ax3.set_xlabel("Confidence")
            ax3.set_ylabel("Accuracy")
            ax3.set_title(f"Calibration (ECE: {cal['ece']:.4f})")
            ax3.legend()
            ax3.grid(alpha=0.3)

        # 4. Overall metrics summary
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis("off")

        summary_text = f"""
        Overall Performance Summary:

        Accuracy: {results['accuracy']:.2%}    |    F1 (macro): {results['f1_macro']:.2%}    |    Top-3 Accuracy: {results['top3_accuracy']:.2%}

        Latency: {results['latency']['mean']:.2f}ms (mean)    |    P95: {results['latency']['p95']:.2f}ms

        Samples: {results['num_samples']}    |    Tools: {len(results['confusion_matrix_labels'])}
        """

        ax4.text(0.5, 0.5, summary_text, ha="center", va="center", fontsize=11, family="monospace")

        fig.suptitle("Routing Evaluation Dashboard", fontsize=16, fontweight="bold")

        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Saved evaluation dashboard to {save_path}")

        self.plt.show()
