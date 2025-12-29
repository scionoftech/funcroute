"""Tests for DataFormatter"""

import pytest
from funcroute.data.formatter import DataFormatter
from funcroute.core.config import ToolDefinition


def test_formatter_init():
    """Test formatter initialization"""
    formatter = DataFormatter()
    assert formatter is not None


def test_extract_tools_from_data():
    """Test extracting unique tool names from data"""
    formatter = DataFormatter()

    data = [
        {"query": "Where is my order?", "tool": "manage_order"},
        {"query": "Track package", "tool": "manage_order"},
        {"query": "Show laptops", "tool": "search_products"},
        {"query": "Find shoes", "tool": "search_products"},
        {"query": "Return item", "tool": "process_return"},
    ]

    tools = formatter.extract_tools_from_data(data)

    assert len(tools) == 3
    assert "manage_order" in tools
    assert "search_products" in tools
    assert "process_return" in tools
    assert tools == sorted(tools)  # Should be sorted


def test_create_tool_definitions_from_names():
    """Test creating simple tool definitions"""
    formatter = DataFormatter()

    tool_names = ["manage_order", "search_products"]
    tools = formatter.create_tool_definitions_from_names(tool_names)

    assert len(tools) == 2
    assert all(isinstance(t, ToolDefinition) for t in tools)
    assert tools[0].name == "manage_order"
    assert tools[1].name == "search_products"


def test_build_tool_definitions_block():
    """Test building FunctionGemma tool definitions block"""
    formatter = DataFormatter()

    tools = [
        ToolDefinition(
            name="manage_order",
            signature="manage_order(order_id: str) -> dict",
            description="Track orders"
        ),
        ToolDefinition(
            name="search_products",
            signature="search_products(query: str) -> list",
            description="Search products"
        ),
    ]

    block = formatter._build_tool_definitions_block(tools)

    # Check structure
    assert "<start_function_declaration>" in block
    assert "<end_function_declaration>" in block
    assert "manage_order(order_id: str) -> dict:" in block
    assert "Track orders" in block
    assert "search_products(query: str) -> list:" in block
    assert "Search products" in block

    # Check NO double blank lines between functions
    assert "\n\n\n" not in block


def test_format_single_sample():
    """Test formatting a single sample"""
    formatter = DataFormatter()

    tools = [
        ToolDefinition(
            name="manage_order",
            signature="manage_order() -> dict",
            description="Track orders"
        ),
    ]

    tool_block = formatter._build_tool_definitions_block(tools)
    sample = {"query": "Where is my order?", "tool": "manage_order"}

    formatted = formatter._format_single_sample(sample, tool_block)

    # Check required tokens
    assert "<start_of_turn>user" in formatted
    assert "User query: Where is my order?" in formatted
    assert "<end_of_turn>" in formatted
    assert "<start_of_turn>model" in formatted
    assert "<function_call>manage_order</function_call>" in formatted

    # Check newline after model
    assert "<start_of_turn>model\n<function_call>" in formatted


def test_format_for_training():
    """Test full training data formatting"""
    formatter = DataFormatter()

    tools = [
        ToolDefinition(
            name="manage_order",
            signature="manage_order() -> dict",
            description="Track orders"
        ),
        ToolDefinition(
            name="search_products",
            signature="search_products() -> list",
            description="Search products"
        ),
    ]

    samples = [
        {"query": "Where is my order?", "tool": "manage_order"},
        {"query": "Show laptops", "tool": "search_products"},
    ]

    formatted = formatter.format_for_training(samples, tools)

    assert len(formatted) == 2
    assert all("text" in item for item in formatted)
    assert all(formatter.validate_format(item["text"]) for item in formatted)


def test_validate_format():
    """Test format validation"""
    formatter = DataFormatter()

    # Valid format
    valid = """<start_of_turn>user
<start_function_declaration>
test() -> dict:
    Test function
<end_function_declaration>

User query: test<end_of_turn>
<start_of_turn>model
<function_call>test</function_call><end_of_turn>"""

    assert formatter.validate_format(valid) is True

    # Invalid: missing tokens
    invalid = "Just some text"
    assert formatter.validate_format(invalid) is False

    # Invalid: wrong model turn format
    invalid_model = """<start_of_turn>user
<start_function_declaration>
test() -> dict:
    Test
<end_function_declaration>

User query: test<end_of_turn>
<start_of_turn>model<function_call>test</function_call><end_of_turn>"""

    assert formatter.validate_format(invalid_model) is False
