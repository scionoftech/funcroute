"""Core FuncRoute functionality"""

# Lazy imports to avoid requiring torch when only using config
def __getattr__(name):
    """Lazy loading to avoid heavy dependencies"""
    if name == "FuncRoute":
        from funcroute.core.router import FuncRoute
        return FuncRoute
    elif name in ("TrainingConfig", "InferenceConfig", "SyntheticDataConfig", "RouteResult", "ToolDefinition"):
        from funcroute.core.config import (
            TrainingConfig,
            InferenceConfig,
            SyntheticDataConfig,
            RouteResult,
            ToolDefinition,
        )
        return locals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "FuncRoute",
    "TrainingConfig",
    "InferenceConfig",
    "SyntheticDataConfig",
    "RouteResult",
    "ToolDefinition",
]
