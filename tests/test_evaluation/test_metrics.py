"""Tests for evaluation metrics"""

import pytest
import numpy as np
from funcroute.evaluation import metrics


def test_accuracy_perfect():
    """Test accuracy with perfect predictions"""
    preds = ["tool1", "tool2", "tool3"]
    truth = ["tool1", "tool2", "tool3"]
    acc = metrics.accuracy(preds, truth)
    assert acc == 1.0


def test_accuracy_zero():
    """Test accuracy with all wrong predictions"""
    preds = ["tool1", "tool1", "tool1"]
    truth = ["tool2", "tool2", "tool2"]
    acc = metrics.accuracy(preds, truth)
    assert acc == 0.0


def test_accuracy_partial():
    """Test accuracy with partial correctness"""
    preds = ["tool1", "tool2", "tool1"]
    truth = ["tool1", "tool1", "tool1"]
    acc = metrics.accuracy(preds, truth)
    assert abs(acc - 0.666666) < 0.001


def test_accuracy_empty():
    """Test accuracy with empty inputs"""
    acc = metrics.accuracy([], [])
    assert acc == 0.0


def test_accuracy_mismatched_length():
    """Test accuracy with mismatched lengths"""
    with pytest.raises(ValueError, match="same length"):
        metrics.accuracy(["tool1"], ["tool1", "tool2"])


def test_precision_recall_f1_perfect():
    """Test PRF with perfect predictions"""
    preds = ["tool1", "tool2", "tool3"]
    truth = ["tool1", "tool2", "tool3"]

    p, r, f1 = metrics.precision_recall_f1(preds, truth, average="macro")
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


def test_precision_recall_f1_macro():
    """Test macro-averaged PRF"""
    preds = ["tool1", "tool2", "tool1", "tool3"]
    truth = ["tool1", "tool1", "tool1", "tool3"]

    p, r, f1 = metrics.precision_recall_f1(preds, truth, average="macro")

    # Should have reasonable values
    assert 0 <= p <= 1
    assert 0 <= r <= 1
    assert 0 <= f1 <= 1


def test_precision_recall_f1_micro():
    """Test micro-averaged PRF"""
    preds = ["tool1", "tool2", "tool1"]
    truth = ["tool1", "tool1", "tool1"]

    p, r, f1 = metrics.precision_recall_f1(preds, truth, average="micro")

    # Micro accuracy should equal overall accuracy
    acc = metrics.accuracy(preds, truth)
    assert abs(p - acc) < 0.001
    assert abs(r - acc) < 0.001


def test_precision_recall_f1_weighted():
    """Test weighted PRF"""
    preds = ["tool1", "tool2", "tool1"]
    truth = ["tool1", "tool1", "tool1"]

    p, r, f1 = metrics.precision_recall_f1(preds, truth, average="weighted")

    assert 0 <= p <= 1
    assert 0 <= r <= 1
    assert 0 <= f1 <= 1


def test_precision_recall_f1_invalid_average():
    """Test invalid averaging method"""
    with pytest.raises(ValueError, match="Unknown average method"):
        metrics.precision_recall_f1(["tool1"], ["tool1"], average="invalid")


def test_per_tool_metrics():
    """Test per-tool metrics calculation"""
    preds = ["tool1", "tool2", "tool1", "tool1"]
    truth = ["tool1", "tool1", "tool1", "tool3"]

    result = metrics.per_tool_metrics(preds, truth)

    # Should have metrics for all tools
    assert "tool1" in result
    assert "tool2" in result
    assert "tool3" in result

    # Check structure
    for tool, tool_metrics in result.items():
        assert "precision" in tool_metrics
        assert "recall" in tool_metrics
        assert "f1" in tool_metrics
        assert "accuracy" in tool_metrics
        assert "support" in tool_metrics
        assert "tp" in tool_metrics
        assert "fp" in tool_metrics
        assert "fn" in tool_metrics
        assert "tn" in tool_metrics

    # tool1 has 3 true instances, 3 predictions, 2 correct
    assert result["tool1"]["support"] == 3


def test_confusion_matrix():
    """Test confusion matrix creation"""
    preds = ["tool1", "tool2", "tool1"]
    truth = ["tool1", "tool1", "tool2"]

    matrix, labels = metrics.confusion_matrix(preds, truth)

    # Should have 2 tools
    assert len(labels) == 2
    assert "tool1" in labels
    assert "tool2" in labels

    # Matrix should be 2x2
    assert matrix.shape == (2, 2)

    # Total should equal number of samples
    assert matrix.sum() == 3


def test_confusion_matrix_single_class():
    """Test confusion matrix with single class"""
    preds = ["tool1", "tool1", "tool1"]
    truth = ["tool1", "tool1", "tool1"]

    matrix, labels = metrics.confusion_matrix(preds, truth)

    assert len(labels) == 1
    assert matrix.shape == (1, 1)
    assert matrix[0, 0] == 3


def test_confidence_calibration():
    """Test confidence calibration"""
    # Perfect calibration
    confs = [0.9, 0.8, 0.7, 0.6]
    preds = ["tool1", "tool1", "tool1", "tool1"]
    truth = ["tool1", "tool1", "tool1", "tool1"]

    cal = metrics.confidence_calibration(confs, preds, truth, num_bins=5)

    assert "ece" in cal
    assert "bin_accuracies" in cal
    assert "bin_confidences" in cal
    assert "bin_counts" in cal

    # Perfect predictions should have low ECE
    assert cal["ece"] >= 0


def test_confidence_calibration_poor():
    """Test calibration with poorly calibrated model"""
    # High confidence but wrong
    confs = [0.9, 0.9, 0.9, 0.9]
    preds = ["tool1", "tool1", "tool1", "tool1"]
    truth = ["tool2", "tool2", "tool2", "tool2"]

    cal = metrics.confidence_calibration(confs, preds, truth, num_bins=5)

    # Should have high ECE (poor calibration)
    assert cal["ece"] > 0


def test_latency_stats():
    """Test latency statistics"""
    latencies = [10.0, 12.0, 15.0, 11.0, 13.0]

    stats = metrics.latency_stats(latencies)

    assert "mean" in stats
    assert "median" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats
    assert "p50" in stats
    assert "p90" in stats
    assert "p95" in stats
    assert "p99" in stats

    assert stats["min"] == 10.0
    assert stats["max"] == 15.0
    assert stats["median"] == 12.0


def test_latency_stats_empty():
    """Test latency stats with empty input"""
    stats = metrics.latency_stats([])

    assert stats["mean"] == 0.0
    assert stats["median"] == 0.0


def test_top_k_accuracy_perfect():
    """Test top-k accuracy with perfect predictions"""
    preds = [
        [("tool1", 0.9), ("tool2", 0.1)],
        [("tool2", 0.8), ("tool1", 0.2)],
    ]
    truth = ["tool1", "tool2"]

    acc = metrics.top_k_accuracy(preds, truth, k=1)
    assert acc == 1.0


def test_top_k_accuracy_top3():
    """Test top-3 accuracy"""
    preds = [
        [("tool1", 0.5), ("tool2", 0.3), ("tool3", 0.2)],
        [("tool1", 0.5), ("tool2", 0.3), ("tool3", 0.2)],
    ]
    truth = ["tool2", "tool3"]  # Both in top-3

    acc = metrics.top_k_accuracy(preds, truth, k=3)
    assert acc == 1.0


def test_top_k_accuracy_partial():
    """Test top-k with partial correctness"""
    preds = [
        [("tool1", 0.9), ("tool2", 0.1)],
        [("tool1", 0.9), ("tool2", 0.1)],
    ]
    truth = ["tool1", "tool3"]  # Only first is in top-2

    acc = metrics.top_k_accuracy(preds, truth, k=2)
    assert acc == 0.5


def test_top_k_accuracy_empty():
    """Test top-k with empty input"""
    acc = metrics.top_k_accuracy([], [], k=1)
    assert acc == 0.0
