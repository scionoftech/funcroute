"""
FuncRoute - Intelligent Function/Tool Routing using FunctionGemma

A Python package for smart task routing in agentic AI systems.
"""

__version__ = "0.1.0"
__author__ = "FuncRoute Contributors"
__license__ = "MIT"

# Lazy imports to avoid requiring torch for data pipeline usage
def __getattr__(name):
    """Lazy loading of modules to avoid heavy dependencies on import"""
    if name == "FuncRoute":
        from funcroute.core.router import FuncRoute
        return FuncRoute
    elif name in ("TrainingConfig", "InferenceConfig", "SyntheticDataConfig", "RouteResult", "ToolDefinition"):
        from funcroute.core import config
        return getattr(config, name)
    elif name == "SyntheticDataGenerator":
        from funcroute.data.generator import SyntheticDataGenerator
        return SyntheticDataGenerator
    elif name == "DataLoader":
        from funcroute.data.loader import DataLoader
        return DataLoader
    elif name == "DataFormatter":
        from funcroute.data.formatter import DataFormatter
        return DataFormatter
    elif name == "Evaluator":
        from funcroute.evaluation.evaluator import Evaluator
        return Evaluator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "FuncRoute",
    "TrainingConfig",
    "InferenceConfig",
    "SyntheticDataConfig",
    "RouteResult",
    "ToolDefinition",
    "SyntheticDataGenerator",
    "DataLoader",
    "DataFormatter",
    "Evaluator",
]
