"""Tests for DataValidator"""

import pytest
from funcroute.data.validator import DataValidator


def test_validator_init():
    """Test validator initialization"""
    validator = DataValidator()
    assert validator is not None


def test_validate_basic():
    """Test basic validation with good data"""
    validator = DataValidator()

    data = [
        {"query": "Where is my order?", "tool": "manage_order"},
        {"query": "Track my package", "tool": "manage_order"},
        {"query": "Show me laptops", "tool": "search_products"},
        {"query": "Find shoes", "tool": "search_products"},
        {"query": "Return this item", "tool": "process_return"},
        {"query": "Get a refund", "tool": "process_return"},
    ]

    report = validator.validate(data, min_samples_per_tool=2)

    assert report["is_valid"] is True
    assert "stats" in report
    assert report["stats"]["num_samples"] == 6
    assert report["stats"]["num_tools"] == 3


def test_validate_empty_data():
    """Test validation with empty data"""
    validator = DataValidator()

    report = validator.validate([])

    assert report["is_valid"] is False
    assert "Data is empty" in report["errors"]


def test_validate_min_samples_warning():
    """Test warning when tool has too few samples"""
    validator = DataValidator()

    data = [
        {"query": "Q1", "tool": "tool1"},
        {"query": "Q2", "tool": "tool1"},
        {"query": "Q3", "tool": "tool2"},  # Only 1 sample!
    ]

    report = validator.validate(data, min_samples_per_tool=5)

    assert len(report["warnings"]) > 0
    assert any("tool1" in w for w in report["warnings"])
    assert any("tool2" in w for w in report["warnings"])


def test_validate_duplicates():
    """Test duplicate query detection"""
    validator = DataValidator()

    data = [
        {"query": "Where is my order?", "tool": "manage_order"},
        {"query": "Where is my order?", "tool": "manage_order"},  # Duplicate!
        {"query": "Track package", "tool": "manage_order"},
    ]

    report = validator.validate(data, warn_duplicates=True)

    assert len(report["warnings"]) > 0
    assert any("duplicate" in w.lower() for w in report["warnings"])


def test_validate_no_duplicate_warning_when_disabled():
    """Test that duplicate warning can be disabled"""
    validator = DataValidator()

    data = [
        {"query": "Same query", "tool": "tool1"},
        {"query": "Same query", "tool": "tool1"},
    ]

    report = validator.validate(data, warn_duplicates=False)

    # Should not have duplicate warning
    assert not any("duplicate" in w.lower() for w in report["warnings"])


def test_validate_class_imbalance():
    """Test class imbalance detection"""
    validator = DataValidator()

    data = []
    # Create imbalanced data: 30 vs 10 vs 5
    for i in range(30):
        data.append({"query": f"Q{i}", "tool": "tool1"})
    for i in range(10):
        data.append({"query": f"Q{i+30}", "tool": "tool2"})
    for i in range(5):
        data.append({"query": f"Q{i+40}", "tool": "tool3"})

    report = validator.validate(data, warn_imbalance=True)

    # 30/5 = 6:1 ratio, should warn (threshold is 3:1)
    assert len(report["warnings"]) > 0
    assert any("imbalance" in w.lower() for w in report["warnings"])


def test_validate_no_imbalance_warning_when_disabled():
    """Test that imbalance warning can be disabled"""
    validator = DataValidator()

    data = []
    for i in range(30):
        data.append({"query": f"Q{i}", "tool": "tool1"})
    for i in range(5):
        data.append({"query": f"Q{i+30}", "tool": "tool2"})

    report = validator.validate(data, warn_imbalance=False)

    # Should not have imbalance warning
    assert not any("imbalance" in w.lower() for w in report["warnings"])


def test_validate_query_length_stats():
    """Test query length statistics"""
    validator = DataValidator()

    data = [
        {"query": "Q1", "tool": "t1"},  # 2 chars
        {"query": "Query 2", "tool": "t1"},  # 7 chars
        {"query": "This is a longer query", "tool": "t1"},  # 22 chars
    ]

    report = validator.validate(data)

    assert "avg_query_length" in report["stats"]
    assert "min_query_length" in report["stats"]
    assert "max_query_length" in report["stats"]

    assert report["stats"]["min_query_length"] == 2
    assert report["stats"]["max_query_length"] == 22
    assert report["stats"]["avg_query_length"] > 0


def test_validate_short_queries_warning():
    """Test warning for very short queries"""
    validator = DataValidator()

    data = [
        {"query": "Q", "tool": "t1"},  # 1 char - too short!
        {"query": "AB", "tool": "t1"},  # 2 chars - too short!
        {"query": "Normal query", "tool": "t1"},
    ]

    report = validator.validate(data)

    assert len(report["warnings"]) > 0
    assert any("short queries" in w.lower() for w in report["warnings"])


def test_validate_tool_counts():
    """Test tool count statistics"""
    validator = DataValidator()

    data = [
        {"query": "Q1", "tool": "tool1"},
        {"query": "Q2", "tool": "tool1"},
        {"query": "Q3", "tool": "tool1"},
        {"query": "Q4", "tool": "tool2"},
        {"query": "Q5", "tool": "tool2"},
    ]

    report = validator.validate(data)

    assert "tool_counts" in report["stats"]
    assert report["stats"]["tool_counts"]["tool1"] == 3
    assert report["stats"]["tool_counts"]["tool2"] == 2


def test_check_leakage_no_overlap():
    """Test leakage check with no overlap"""
    validator = DataValidator()

    train_data = [
        {"query": "Train Q1", "tool": "t1"},
        {"query": "Train Q2", "tool": "t1"},
    ]

    test_data = [
        {"query": "Test Q1", "tool": "t1"},
        {"query": "Test Q2", "tool": "t1"},
    ]

    result = validator.check_leakage(train_data, test_data)

    assert result is True


def test_check_leakage_with_overlap():
    """Test leakage check with overlap"""
    validator = DataValidator()

    train_data = [
        {"query": "Shared query", "tool": "t1"},  # Overlaps!
        {"query": "Train Q2", "tool": "t1"},
    ]

    test_data = [
        {"query": "Shared query", "tool": "t1"},  # Overlaps!
        {"query": "Test Q2", "tool": "t1"},
    ]

    result = validator.check_leakage(train_data, test_data)

    assert result is False


def test_print_report(capsys):
    """Test print_report output"""
    validator = DataValidator()

    data = [
        {"query": "Q1", "tool": "tool1"},
        {"query": "Q2", "tool": "tool1"},
        {"query": "Q3", "tool": "tool2"},
    ]

    report = validator.validate(data)
    validator.print_report(report)

    captured = capsys.readouterr()

    # Check that key sections are printed
    assert "DATA VALIDATION REPORT" in captured.out
    assert "Dataset Statistics" in captured.out
    assert "Tool Distribution" in captured.out


def test_validate_all_clean():
    """Test validation with perfect data"""
    validator = DataValidator()

    data = []
    for i in range(20):
        data.append({"query": f"Query {i} for tool1", "tool": "tool1"})
    for i in range(20):
        data.append({"query": f"Query {i} for tool2", "tool": "tool2"})
    for i in range(20):
        data.append({"query": f"Query {i} for tool3", "tool": "tool3"})

    report = validator.validate(data, min_samples_per_tool=10)

    # Should be valid with no warnings
    assert report["is_valid"] is True
    assert len(report["errors"]) == 0
    assert len(report["warnings"]) == 0


def test_validate_multiple_issues():
    """Test validation with multiple issues"""
    validator = DataValidator()

    data = []
    # Create data with multiple problems:
    # 1. Imbalanced (30 vs 3)
    # 2. Duplicates
    # 3. Short queries
    for i in range(30):
        data.append({"query": "Q", "tool": "tool1"})  # Short!
    for i in range(3):
        data.append({"query": "AB", "tool": "tool2"})  # Short!
    data.append({"query": "Q", "tool": "tool1"})  # Duplicate!

    report = validator.validate(data, min_samples_per_tool=10)

    # Should have multiple warnings
    assert len(report["warnings"]) >= 2

    # Check for specific warning types
    warning_text = " ".join(report["warnings"]).lower()
    assert "imbalance" in warning_text or "short" in warning_text or "duplicate" in warning_text
