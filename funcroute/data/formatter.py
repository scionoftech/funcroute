"""
FunctionGemma data formatting

Converts simple query-tool pairs to FunctionGemma's required format.
Based on train.py lines 709-746.
"""

from typing import List, Dict, Any
from funcroute.core.config import ToolDefinition


class DataFormatter:
    """Format training data for FunctionGemma"""

    def __init__(self):
        """Initialize formatter"""
        pass

    def format_for_training(
        self,
        samples: List[Dict[str, str]],
        tools: List[ToolDefinition]
    ) -> List[Dict[str, str]]:
        """
        Convert training samples to FunctionGemma format.

        Args:
            samples: List of {"query": str, "tool": str} dicts
            tools: List of all available ToolDefinitions

        Returns:
            List of {"text": formatted_string} for training

        Example:
            >>> samples = [{"query": "Where is my order?", "tool": "manage_order"}]
            >>> tools = [ToolDefinition(name="manage_order", ...)]
            >>> formatter = DataFormatter()
            >>> formatted = formatter.format_for_training(samples, tools)
        """
        formatted_samples = []

        # Build tool definitions block once (same for all samples)
        tool_definitions_block = self._build_tool_definitions_block(tools)

        for sample in samples:
            formatted_text = self._format_single_sample(
                sample,
                tool_definitions_block
            )
            formatted_samples.append({"text": formatted_text})

        return formatted_samples

    def _build_tool_definitions_block(self, tools: List[ToolDefinition]) -> str:
        """
        Build the <start_function_declaration> block with ALL tools.

        CRITICAL RULES (from train.py lines 720-735):
        1. NO blank lines between function definitions
        2. Format: signature:\\n    description (4 spaces, no blank line)
        3. All tools must be listed

        Args:
            tools: List of ToolDefinitions

        Returns:
            Complete function declaration block as string
        """
        # Start declaration block
        lines = ["<start_function_declaration>"]

        # Add each tool (NO blank lines between them!)
        for tool in tools:
            # Add signature
            lines.append(f"{tool.signature}:")
            # Add description (4 spaces indentation, no blank line)
            lines.append(f"    {tool.description}")

        # End declaration block
        lines.append("<end_function_declaration>")

        return "\n".join(lines)

    def _format_single_sample(
        self,
        sample: Dict[str, str],
        tool_definitions_block: str
    ) -> str:
        """
        Format a single sample with FunctionGemma specification.

        CRITICAL RULES (from train.py lines 738-743):
        1. Newline AFTER <start_of_turn>model
        2. Exact format structure

        Args:
            sample: {"query": str, "tool": str}
            tool_definitions_block: Pre-built tool definitions

        Returns:
            Formatted training text
        """
        formatted = f"""<start_of_turn>user
{tool_definitions_block}

User query: {sample['query']}<end_of_turn>
<start_of_turn>model
<function_call>{sample['tool']}</function_call><end_of_turn>"""

        return formatted

    def validate_format(self, formatted_text: str) -> bool:
        """
        Validate that formatted text follows FunctionGemma rules.

        Args:
            formatted_text: Formatted training sample

        Returns:
            True if valid, False otherwise
        """
        required_tokens = [
            "<start_of_turn>user",
            "<start_function_declaration>",
            "<end_function_declaration>",
            "User query:",
            "<end_of_turn>",
            "<start_of_turn>model",
            "<function_call>",
            "</function_call>",
        ]

        # Check all required tokens present
        for token in required_tokens:
            if token not in formatted_text:
                return False

        # Check correct model turn format (newline after model)
        if "<start_of_turn>model\n<function_call>" not in formatted_text:
            return False

        # Check no double newline before model
        if "\n\n<start_of_turn>model" in formatted_text:
            return False

        return True

    def extract_tools_from_data(self, samples: List[Dict[str, str]]) -> List[str]:
        """
        Extract unique tool names from training data.

        Args:
            samples: List of {"query": str, "tool": str}

        Returns:
            List of unique tool names, sorted
        """
        tool_names = set()
        for sample in samples:
            if "tool" in sample:
                tool_names.add(sample["tool"])

        return sorted(list(tool_names))

    def create_tool_definitions_from_names(
        self,
        tool_names: List[str]
    ) -> List[ToolDefinition]:
        """
        Create simple ToolDefinitions from tool names.

        Used when user doesn't provide explicit ToolDefinitions.

        Args:
            tool_names: List of tool name strings

        Returns:
            List of ToolDefinitions with basic info
        """
        tool_definitions = []

        for name in tool_names:
            # Create simple definition
            tool_def = ToolDefinition(
                name=name,
                signature=f"{name}() -> dict",
                description=f"Handles {name} queries",
                parameters=None,
                examples=[],
                keywords=[]
            )
            tool_definitions.append(tool_def)

        return tool_definitions
