"""Tests for SyntheticDataGenerator"""

import pytest
from funcroute.data.generator import SyntheticDataGenerator
from funcroute.core.config import ToolDefinition


def test_generator_init_rule_based():
    """Test generator initialization with rule-based method"""
    generator = SyntheticDataGenerator(method="rule_based")
    assert generator is not None
    assert generator.method == "rule_based"


def test_generator_init_llm_based():
    """Test generator initialization with LLM-based method"""
    generator = SyntheticDataGenerator(method="llm_based", llm_model="gpt2")
    assert generator is not None
    assert generator.method == "llm_based"
    assert generator.llm_model == "gpt2"


def test_rule_based_generation():
    """Test rule-based synthetic data generation"""
    generator = SyntheticDataGenerator(method="rule_based")

    tools = [
        ToolDefinition(
            name="manage_order",
            signature="manage_order(order_id: str) -> dict",
            description="Track and manage customer orders",
            examples=["Where is my order?", "Track package"]
        ),
        ToolDefinition(
            name="search_products",
            signature="search_products(query: str) -> list",
            description="Search for products",
            examples=["Find laptops", "Show me shoes"]
        ),
    ]

    data = generator.generate(tools, num_variations=10, num_samples=50)

    # Check output format
    assert isinstance(data, list)
    assert len(data) > 0
    assert all("query" in item for item in data)
    assert all("tool" in item for item in data)

    # Check that we have both tools
    tools_used = set(item["tool"] for item in data)
    assert "manage_order" in tools_used
    assert "search_products" in tools_used


def test_rule_based_variations():
    """Test that rule-based generation creates variations"""
    generator = SyntheticDataGenerator(method="rule_based")

    tools = [
        ToolDefinition(
            name="test_tool",
            signature="test_tool() -> str",
            description="Test tool",
            examples=["test query"]
        ),
    ]

    data = generator.generate(tools, num_variations=20, num_samples=100)

    # Should have multiple variations
    queries = [item["query"] for item in data]
    unique_queries = set(queries)

    # Should have more than just the base example
    assert len(unique_queries) > 1


def test_create_pattern_group():
    """Test pattern group creation with variations"""
    generator = SyntheticDataGenerator(method="rule_based")

    base_pattern = "Where is my order?"
    tool = "manage_order"
    num_variations = 50

    pattern_group = generator._create_pattern_group(base_pattern, tool, num_variations)

    # Check structure
    assert "base_pattern" in pattern_group
    assert "tool" in pattern_group
    assert "variations" in pattern_group

    assert pattern_group["base_pattern"] == base_pattern
    assert pattern_group["tool"] == tool
    assert len(pattern_group["variations"]) <= num_variations


def test_generator_has_variation_templates():
    """Test that generator has variation templates"""
    generator = SyntheticDataGenerator(method="rule_based")

    # Check that variation templates are loaded
    assert hasattr(generator, 'prefixes')
    assert hasattr(generator, 'suffixes')
    assert hasattr(generator, 'contexts')
    assert hasattr(generator, 'time_markers')

    assert len(generator.prefixes) > 0
    assert len(generator.suffixes) > 0
    assert len(generator.contexts) > 0
    assert len(generator.time_markers) > 0


def test_invalid_method():
    """Test invalid generation method"""
    with pytest.raises(ValueError, match="Unknown method"):
        generator = SyntheticDataGenerator(method="invalid_method")
        generator.generate([], num_samples=10)


def test_no_examples():
    """Test generation with no examples provided"""
    generator = SyntheticDataGenerator(method="rule_based")

    tools = [
        ToolDefinition(
            name="test_tool",
            signature="test_tool() -> str",
            description="Test tool",
            examples=[]  # No examples!
        ),
    ]

    # Should handle gracefully - either skip or use tool name
    data = generator.generate(tools, num_variations=5, num_samples=10)
    assert isinstance(data, list)


def test_keywords_usage():
    """Test that keywords are used in generation"""
    generator = SyntheticDataGenerator(method="rule_based")

    tools = [
        ToolDefinition(
            name="search_tool",
            signature="search_tool() -> list",
            description="Search",
            examples=["find items"],
            keywords=["search", "find", "look for", "query"]
        ),
    ]

    data = generator.generate(tools, num_variations=20, num_samples=50)

    # Should have variations using keywords
    all_queries = " ".join(item["query"].lower() for item in data)
    # At least some keywords should appear
    assert any(kw in all_queries for kw in ["search", "find", "look for", "query"])


@pytest.mark.skip(reason="Requires HuggingFace model download - slow test")
def test_llm_based_generation():
    """Test LLM-based generation (requires model download)"""
    generator = SyntheticDataGenerator(method="llm_based", llm_model="gpt2")

    tools = [
        ToolDefinition(
            name="manage_order",
            signature="manage_order() -> dict",
            description="Manage orders",
            examples=["Where is my order?"]
        ),
    ]

    data = generator.generate(tools, num_samples=10)

    assert isinstance(data, list)
    assert len(data) > 0
    assert all("query" in item for item in data)
    assert all("tool" in item for item in data)
