"""
Core evaluation metrics for routing performance
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import Counter


def accuracy(predictions: List[str], ground_truth: List[str]) -> float:
    """
    Calculate routing accuracy.

    Args:
        predictions: List of predicted tool names
        ground_truth: List of actual tool names

    Returns:
        Accuracy score (0-1)

    Example:
        >>> preds = ["tool1", "tool2", "tool1"]
        >>> truth = ["tool1", "tool1", "tool1"]
        >>> accuracy(preds, truth)
        0.6666666666666666
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    if len(predictions) == 0:
        return 0.0

    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def precision_recall_f1(
    predictions: List[str],
    ground_truth: List[str],
    average: str = "macro",
) -> Tuple[float, float, float]:
    """
    Calculate precision, recall, and F1 score.

    Args:
        predictions: List of predicted tool names
        ground_truth: List of actual tool names
        average: "macro" (default), "micro", or "weighted"

    Returns:
        (precision, recall, f1) tuple

    Example:
        >>> preds = ["tool1", "tool2", "tool1"]
        >>> truth = ["tool1", "tool1", "tool1"]
        >>> p, r, f1 = precision_recall_f1(preds, truth)
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    # Get unique tools
    tools = sorted(set(ground_truth) | set(predictions))

    if len(tools) == 0:
        return 0.0, 0.0, 0.0

    # Calculate per-tool metrics
    tool_metrics = {}
    for tool in tools:
        # True positives: predicted and actual are both this tool
        tp = sum((p == tool and g == tool) for p, g in zip(predictions, ground_truth))

        # False positives: predicted this tool but actual is different
        fp = sum((p == tool and g != tool) for p, g in zip(predictions, ground_truth))

        # False negatives: actual is this tool but predicted different
        fn = sum((p != tool and g == tool) for p, g in zip(predictions, ground_truth))

        # Per-tool precision and recall
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        tool_metrics[tool] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sum(g == tool for g in ground_truth),
        }

    # Aggregate based on averaging method
    if average == "macro":
        # Simple average across tools
        precision = np.mean([m["precision"] for m in tool_metrics.values()])
        recall = np.mean([m["recall"] for m in tool_metrics.values()])
        f1 = np.mean([m["f1"] for m in tool_metrics.values()])

    elif average == "micro":
        # Aggregate all TP, FP, FN then calculate
        total_tp = sum(
            sum((p == tool and g == tool) for p, g in zip(predictions, ground_truth))
            for tool in tools
        )
        total_fp = sum(
            sum((p == tool and g != tool) for p, g in zip(predictions, ground_truth))
            for tool in tools
        )
        total_fn = sum(
            sum((p != tool and g == tool) for p, g in zip(predictions, ground_truth))
            for tool in tools
        )

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    elif average == "weighted":
        # Weighted by support (number of true instances per tool)
        total_support = sum(m["support"] for m in tool_metrics.values())

        if total_support == 0:
            return 0.0, 0.0, 0.0

        precision = sum(
            m["precision"] * m["support"] for m in tool_metrics.values()
        ) / total_support
        recall = sum(
            m["recall"] * m["support"] for m in tool_metrics.values()
        ) / total_support
        f1 = sum(m["f1"] * m["support"] for m in tool_metrics.values()) / total_support

    else:
        raise ValueError(f"Unknown average method: {average}")

    return precision, recall, f1


def per_tool_metrics(
    predictions: List[str],
    ground_truth: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics for each tool individually.

    Args:
        predictions: List of predicted tool names
        ground_truth: List of actual tool names

    Returns:
        Dict mapping tool name to metrics dict

    Example:
        >>> preds = ["tool1", "tool2", "tool1"]
        >>> truth = ["tool1", "tool1", "tool1"]
        >>> metrics = per_tool_metrics(preds, truth)
        >>> metrics["tool1"]["accuracy"]
        0.6666666666666666
    """
    tools = sorted(set(ground_truth) | set(predictions))
    results = {}

    for tool in tools:
        # True positives, false positives, false negatives
        tp = sum((p == tool and g == tool) for p, g in zip(predictions, ground_truth))
        fp = sum((p == tool and g != tool) for p, g in zip(predictions, ground_truth))
        fn = sum((p != tool and g == tool) for p, g in zip(predictions, ground_truth))
        tn = sum((p != tool and g != tool) for p, g in zip(predictions, ground_truth))

        # Support
        support = sum(g == tool for g in ground_truth)

        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy_score = (tp + tn) / len(predictions) if len(predictions) > 0 else 0.0

        results[tool] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy_score,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    return results


def confusion_matrix(
    predictions: List[str],
    ground_truth: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Create confusion matrix.

    Args:
        predictions: List of predicted tool names
        ground_truth: List of actual tool names

    Returns:
        (matrix, labels) where matrix[i][j] is count of
        true label i predicted as label j

    Example:
        >>> preds = ["tool1", "tool2", "tool1"]
        >>> truth = ["tool1", "tool1", "tool1"]
        >>> matrix, labels = confusion_matrix(preds, truth)
        >>> labels
        ['tool1', 'tool2']
    """
    # Get sorted unique tools
    tools = sorted(set(ground_truth) | set(predictions))
    tool_to_idx = {tool: i for i, tool in enumerate(tools)}

    # Create matrix
    n = len(tools)
    matrix = np.zeros((n, n), dtype=int)

    for pred, true in zip(predictions, ground_truth):
        true_idx = tool_to_idx[true]
        pred_idx = tool_to_idx[pred]
        matrix[true_idx][pred_idx] += 1

    return matrix, tools


def confidence_calibration(
    confidences: List[float],
    predictions: List[str],
    ground_truth: List[str],
    num_bins: int = 10,
) -> Dict[str, any]:
    """
    Evaluate confidence calibration.

    Good calibration means that when the model says it's X% confident,
    it's correct X% of the time.

    Args:
        confidences: List of confidence scores (0-1)
        predictions: List of predicted tool names
        ground_truth: List of actual tool names
        num_bins: Number of bins for calibration

    Returns:
        Dict with calibration metrics

    Example:
        >>> confs = [0.9, 0.7, 0.6]
        >>> preds = ["tool1", "tool2", "tool1"]
        >>> truth = ["tool1", "tool1", "tool1"]
        >>> cal = confidence_calibration(confs, preds, truth)
    """
    if len(confidences) != len(predictions) or len(predictions) != len(ground_truth):
        raise ValueError("All inputs must have same length")

    # Create bins
    bins = np.linspace(0, 1, num_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        # Find samples in this bin
        in_bin = [
            (c >= bin_lower and c < bin_upper) or (i == num_bins - 1 and c == bin_upper)
            for c in confidences
        ]

        if not any(in_bin):
            continue

        # Calculate accuracy in this bin
        bin_preds = [p for p, in_b in zip(predictions, in_bin) if in_b]
        bin_truth = [g for g, in_b in zip(ground_truth, in_bin) if in_b]
        bin_confs = [c for c, in_b in zip(confidences, in_bin) if in_b]

        bin_acc = accuracy(bin_preds, bin_truth)
        bin_conf = np.mean(bin_confs)

        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)
        bin_counts.append(len(bin_preds))

    # Expected Calibration Error (ECE)
    if len(bin_accuracies) > 0:
        total_samples = sum(bin_counts)
        ece = sum(
            (count / total_samples) * abs(acc - conf)
            for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts)
        )
    else:
        ece = 0.0

    return {
        "ece": ece,  # Expected Calibration Error
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "num_bins": num_bins,
    }


def latency_stats(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate latency statistics.

    Args:
        latencies: List of latency values in milliseconds

    Returns:
        Dict with latency statistics

    Example:
        >>> lats = [10.5, 12.3, 9.8, 15.2]
        >>> stats = latency_stats(lats)
        >>> stats["mean"]
        11.95
    """
    if len(latencies) == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    return {
        "mean": float(np.mean(latencies)),
        "median": float(np.median(latencies)),
        "std": float(np.std(latencies)),
        "min": float(np.min(latencies)),
        "max": float(np.max(latencies)),
        "p50": float(np.percentile(latencies, 50)),
        "p90": float(np.percentile(latencies, 90)),
        "p95": float(np.percentile(latencies, 95)),
        "p99": float(np.percentile(latencies, 99)),
    }


def top_k_accuracy(
    predictions_with_scores: List[List[Tuple[str, float]]],
    ground_truth: List[str],
    k: int = 3,
) -> float:
    """
    Calculate top-k accuracy.

    Model is correct if true label is in top-k predictions.

    Args:
        predictions_with_scores: List of [(tool, confidence), ...] per sample
        ground_truth: List of actual tool names
        k: Number of top predictions to consider

    Returns:
        Top-k accuracy score (0-1)

    Example:
        >>> preds = [
        ...     [("tool1", 0.9), ("tool2", 0.1)],
        ...     [("tool2", 0.6), ("tool1", 0.4)],
        ... ]
        >>> truth = ["tool1", "tool1"]
        >>> top_k_accuracy(preds, truth, k=2)
        1.0
    """
    if len(predictions_with_scores) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")

    if len(predictions_with_scores) == 0:
        return 0.0

    correct = 0
    for preds, true_label in zip(predictions_with_scores, ground_truth):
        # Get top-k predicted tools
        top_k_tools = [tool for tool, _ in preds[:k]]

        if true_label in top_k_tools:
            correct += 1

    return correct / len(predictions_with_scores)
