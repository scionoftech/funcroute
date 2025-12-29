"""Tests for Evaluator class"""

import pytest
from funcroute.evaluation.evaluator import Evaluator
from funcroute.core.config import RouteResult


class MockRouter:
    """Mock router for testing"""

    def __init__(self, predictions_map=None):
        """
        Initialize mock router.

        Args:
            predictions_map: Dict mapping query to (tool, confidence)
        """
        self.predictions_map = predictions_map or {}
        self.call_count = 0

    def route(self, query: str) -> RouteResult:
        """Mock routing"""
        self.call_count += 1

        # Get prediction or default
        if query in self.predictions_map:
            tool, confidence = self.predictions_map[query]
        else:
            tool, confidence = "default_tool", 0.5

        return RouteResult(
            query=query,
            tool=tool,
            confidence=confidence,
            latency_ms=10.0 + self.call_count * 0.1,  # Simulated latency
            alternatives=[("alt_tool", 0.3), ("alt_tool2", 0.2)],
        )


def test_evaluator_init():
    """Test evaluator initialization"""
    router = MockRouter()
    evaluator = Evaluator(router)
    assert evaluator.router == router


def test_evaluator_init_no_router():
    """Test evaluator without router"""
    evaluator = Evaluator()
    assert evaluator.router is None


def test_evaluate_perfect_accuracy():
    """Test evaluation with perfect accuracy"""
    # Create mock router that always predicts correctly
    predictions_map = {
        "query1": ("tool1", 0.9),
        "query2": ("tool2", 0.8),
        "query3": ("tool1", 0.95),
    }
    router = MockRouter(predictions_map)

    test_data = [
        {"query": "query1", "tool": "tool1"},
        {"query": "query2", "tool": "tool2"},
        {"query": "query3", "tool": "tool1"},
    ]

    evaluator = Evaluator(router)
    results = evaluator.evaluate(test_data, verbose=False)

    # Check accuracy
    assert results["accuracy"] == 1.0
    assert results["f1_macro"] == 1.0
    assert results["num_samples"] == 3


def test_evaluate_partial_accuracy():
    """Test evaluation with partial accuracy"""
    predictions_map = {
        "query1": ("tool1", 0.9),  # Correct
        "query2": ("tool1", 0.8),  # Wrong (should be tool2)
        "query3": ("tool1", 0.95),  # Correct
    }
    router = MockRouter(predictions_map)

    test_data = [
        {"query": "query1", "tool": "tool1"},
        {"query": "query2", "tool": "tool2"},
        {"query": "query3", "tool": "tool1"},
    ]

    evaluator = Evaluator()
    results = evaluator.evaluate(test_data, router=router, verbose=False)

    # 2/3 correct
    assert abs(results["accuracy"] - 0.666666) < 0.001


def test_evaluate_no_router_error():
    """Test evaluation without router raises error"""
    evaluator = Evaluator()
    test_data = [{"query": "test", "tool": "tool1"}]

    with pytest.raises(ValueError, match="No router provided"):
        evaluator.evaluate(test_data)


def test_evaluate_empty_data_error():
    """Test evaluation with empty data raises error"""
    router = MockRouter()
    evaluator = Evaluator(router)

    with pytest.raises(ValueError, match="Test data is empty"):
        evaluator.evaluate([])


def test_evaluate_includes_all_metrics():
    """Test that evaluate returns all expected metrics"""
    router = MockRouter({"q1": ("tool1", 0.9)})
    test_data = [{"query": "q1", "tool": "tool1"}]

    evaluator = Evaluator(router)
    results = evaluator.evaluate(test_data, verbose=False)

    # Check all required keys
    required_keys = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_micro",
        "recall_micro",
        "f1_micro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "top3_accuracy",
        "top5_accuracy",
        "per_tool_metrics",
        "confusion_matrix",
        "confusion_matrix_labels",
        "calibration",
        "latency",
        "num_samples",
        "predictions",
        "ground_truth",
        "confidences",
    ]

    for key in required_keys:
        assert key in results, f"Missing key: {key}"


def test_evaluate_latency_stats():
    """Test that latency stats are calculated"""
    router = MockRouter({"q1": ("tool1", 0.9)})
    test_data = [{"query": "q1", "tool": "tool1"}]

    evaluator = Evaluator(router)
    results = evaluator.evaluate(test_data, verbose=False)

    latency = results["latency"]
    assert "mean" in latency
    assert "median" in latency
    assert "p90" in latency
    assert "p95" in latency
    assert "p99" in latency

    assert latency["mean"] > 0


def test_evaluate_top_k_accuracy():
    """Test top-k accuracy calculation"""
    router = MockRouter({"q1": ("tool2", 0.9)})  # Wrong prediction
    test_data = [{"query": "q1", "tool": "tool1"}]

    evaluator = Evaluator(router)
    results = evaluator.evaluate(test_data, verbose=False)

    # Top-1 should be 0 (wrong)
    assert results["accuracy"] == 0.0

    # Top-3 might be > 0 if true tool in alternatives
    assert results["top3_accuracy"] >= 0


def test_print_report(capsys):
    """Test print_report output"""
    router = MockRouter({"q1": ("tool1", 0.9)})
    test_data = [{"query": "q1", "tool": "tool1"}]

    evaluator = Evaluator(router)
    results = evaluator.evaluate(test_data, verbose=False)

    evaluator.print_report(results)

    captured = capsys.readouterr()

    # Check key sections are printed
    assert "ROUTING EVALUATION REPORT" in captured.out
    assert "Overall Performance" in captured.out
    assert "Per-Tool Performance" in captured.out
    assert "Latency Statistics" in captured.out
    assert "Confusion Matrix" in captured.out


def test_evaluate_multiple_tools():
    """Test evaluation with multiple tools"""
    predictions_map = {
        "q1": ("tool1", 0.9),
        "q2": ("tool2", 0.8),
        "q3": ("tool3", 0.85),
        "q4": ("tool1", 0.92),
    }
    router = MockRouter(predictions_map)

    test_data = [
        {"query": "q1", "tool": "tool1"},
        {"query": "q2", "tool": "tool2"},
        {"query": "q3", "tool": "tool3"},
        {"query": "q4", "tool": "tool1"},
    ]

    evaluator = Evaluator(router)
    results = evaluator.evaluate(test_data, verbose=False)

    # Should have 3 tools in per_tool_metrics
    assert len(results["per_tool_metrics"]) == 3

    # Confusion matrix should be 3x3
    assert results["confusion_matrix"].shape == (3, 3)


def test_evaluate_calibration():
    """Test confidence calibration is calculated"""
    router = MockRouter({"q1": ("tool1", 0.9), "q2": ("tool2", 0.6)})
    test_data = [{"query": "q1", "tool": "tool1"}, {"query": "q2", "tool": "tool2"}]

    evaluator = Evaluator(router)
    results = evaluator.evaluate(test_data, verbose=False)

    calibration = results["calibration"]
    assert "ece" in calibration
    assert calibration["ece"] >= 0


def test_compare_routers(capsys):
    """Test router comparison"""
    router1 = MockRouter({"q1": ("tool1", 0.9)})
    router2 = MockRouter({"q1": ("tool2", 0.8)})

    routers = {"router1": router1, "router2": router2}
    test_data = [{"query": "q1", "tool": "tool1"}]

    evaluator = Evaluator()
    comparison = evaluator.compare_routers(routers, test_data, verbose=False)

    # Should have results for both routers
    assert "router1" in comparison
    assert "router2" in comparison

    # Router1 should be correct, router2 wrong
    assert comparison["router1"]["accuracy"] == 1.0
    assert comparison["router2"]["accuracy"] == 0.0


def test_compare_routers_with_verbose(capsys):
    """Test router comparison with verbose output"""
    router1 = MockRouter({"q1": ("tool1", 0.9)})
    router2 = MockRouter({"q1": ("tool1", 0.9)})

    routers = {"baseline": router1, "improved": router2}
    test_data = [{"query": "q1", "tool": "tool1"}]

    evaluator = Evaluator()
    comparison = evaluator.compare_routers(routers, test_data, verbose=True)

    captured = capsys.readouterr()

    # Check comparison table is printed
    assert "ROUTER COMPARISON" in captured.out
    assert "baseline" in captured.out
    assert "improved" in captured.out
