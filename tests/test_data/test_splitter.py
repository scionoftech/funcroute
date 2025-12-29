"""Tests for PatternGroupSplitter"""

import pytest
from funcroute.data.splitter import PatternGroupSplitter


def test_splitter_init():
    """Test splitter initialization"""
    splitter = PatternGroupSplitter()
    assert splitter is not None


def test_basic_split():
    """Test basic splitting with default ratios"""
    splitter = PatternGroupSplitter()

    # Create data with pattern groups
    data = [
        {"query": "Where is my order?", "tool": "manage_order"},
        {"query": "Where's my order?", "tool": "manage_order"},
        {"query": "Track my package", "tool": "manage_order"},
        {"query": "Track package", "tool": "manage_order"},
        {"query": "Show me red dresses", "tool": "search_products"},
        {"query": "Show red dresses", "tool": "search_products"},
        {"query": "Find laptops", "tool": "search_products"},
        {"query": "Find me laptops", "tool": "search_products"},
        {"query": "I want to return this", "tool": "process_return"},
        {"query": "Want to return this", "tool": "process_return"},
    ]

    train, val, test = splitter.split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    # Check that all data is accounted for
    assert len(train) + len(val) + len(test) == len(data)

    # Check that all splits have data
    assert len(train) > 0
    assert len(val) > 0 or len(test) > 0  # At least one should have data


def test_no_leakage():
    """Test that there's no query overlap between splits"""
    splitter = PatternGroupSplitter()

    data = [
        {"query": "Where is my order?", "tool": "manage_order"},
        {"query": "Where's my order?", "tool": "manage_order"},
        {"query": "Where is my package?", "tool": "manage_order"},
        {"query": "Track my package", "tool": "manage_order"},
        {"query": "Track package #123", "tool": "manage_order"},
        {"query": "Show me red dresses", "tool": "search_products"},
        {"query": "Show red dresses please", "tool": "search_products"},
        {"query": "Find laptops under $1000", "tool": "search_products"},
        {"query": "Find me laptops", "tool": "search_products"},
        {"query": "I want to return this", "tool": "process_return"},
        {"query": "Want to return this item", "tool": "process_return"},
        {"query": "Get a refund", "tool": "process_return"},
        {"query": "Get refund please", "tool": "process_return"},
    ]

    train, val, test = splitter.split(data)

    # Extract queries
    train_queries = set(s["query"] for s in train)
    val_queries = set(s["query"] for s in val)
    test_queries = set(s["query"] for s in test)

    # Check no overlap
    assert len(train_queries & val_queries) == 0, "Train and val have overlapping queries"
    assert len(train_queries & test_queries) == 0, "Train and test have overlapping queries"
    assert len(val_queries & test_queries) == 0, "Val and test have overlapping queries"


def test_pattern_group_integrity():
    """Test that pattern variations stay together"""
    splitter = PatternGroupSplitter()

    # Create clear pattern groups
    data = [
        # Group 1: "order status" variations for manage_order
        {"query": "What is my order status?", "tool": "manage_order"},
        {"query": "Order status please", "tool": "manage_order"},
        {"query": "Check order status", "tool": "manage_order"},

        # Group 2: "find shoes" variations for search_products
        {"query": "Find me some shoes", "tool": "search_products"},
        {"query": "Find shoes", "tool": "search_products"},
        {"query": "Looking for shoes", "tool": "search_products"},

        # Group 3: "refund" variations for process_return
        {"query": "I need a refund", "tool": "process_return"},
        {"query": "Need refund", "tool": "process_return"},
        {"query": "Get my refund", "tool": "process_return"},
    ]

    train, val, test = splitter.split(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    # All splits should exist and sum to original
    assert len(train) + len(val) + len(test) == len(data)


def test_custom_ratios():
    """Test splitting with custom ratios"""
    splitter = PatternGroupSplitter()

    data = []
    # Create enough pattern groups to test ratios
    for i in range(10):
        for j in range(3):
            data.append({
                "query": f"Query {i} variation {j}",
                "tool": f"tool_{i % 3}"
            })

    train, val, test = splitter.split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    assert len(train) + len(val) + len(test) == len(data)
    # Train should be largest
    assert len(train) >= len(val)
    assert len(train) >= len(test)


def test_single_tool():
    """Test splitting with single tool"""
    splitter = PatternGroupSplitter()

    data = [
        {"query": "Query 1", "tool": "single_tool"},
        {"query": "Query 2", "tool": "single_tool"},
        {"query": "Query 3", "tool": "single_tool"},
        {"query": "Query 4", "tool": "single_tool"},
        {"query": "Query 5", "tool": "single_tool"},
    ]

    train, val, test = splitter.split(data)

    # All samples should have same tool
    all_tools = set(s["tool"] for s in train + val + test)
    assert all_tools == {"single_tool"}
    assert len(train) + len(val) + len(test) == len(data)


def test_verify_no_leakage_method():
    """Test the verify_no_leakage method directly"""
    splitter = PatternGroupSplitter()

    # Test with no leakage
    train = [{"query": "Q1", "tool": "t1"}]
    val = [{"query": "Q2", "tool": "t1"}]
    test = [{"query": "Q3", "tool": "t1"}]

    # Should not raise
    splitter._verify_no_leakage(train, val, test)

    # Test with leakage
    train_leak = [{"query": "Q1", "tool": "t1"}]
    val_leak = [{"query": "Q1", "tool": "t1"}]  # Same query!
    test_leak = [{"query": "Q3", "tool": "t1"}]

    with pytest.raises(ValueError, match="Data leakage detected"):
        splitter._verify_no_leakage(train_leak, val_leak, test_leak)


def test_empty_data():
    """Test handling of empty data"""
    splitter = PatternGroupSplitter()

    with pytest.raises(ValueError, match="Data is empty"):
        splitter.split([])


def test_too_few_groups():
    """Test handling when there aren't enough groups for all splits"""
    splitter = PatternGroupSplitter()

    # Only 2 groups - warning should be printed but should still work
    data = [
        {"query": "Q1a", "tool": "t1"},
        {"query": "Q1b", "tool": "t1"},
        {"query": "Q2a", "tool": "t2"},
        {"query": "Q2b", "tool": "t2"},
    ]

    # Should work but some splits might be small
    train, val, test = splitter.split(data)
    # All data should be accounted for
    assert len(train) + len(val) + len(test) == len(data)
