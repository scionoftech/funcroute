"""Configuration classes for FuncRoute"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Literal, Dict, Any


@dataclass
class ToolDefinition:
    """Definition of a tool/function for routing"""
    name: str
    signature: str
    description: str
    parameters: Optional[Dict[str, Any]] = None
    examples: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "signature": self.signature,
            "description": self.description,
            "parameters": self.parameters,
            "examples": self.examples,
            "keywords": self.keywords,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolDefinition":
        """Create from dictionary"""
        return cls(
            name=data["name"],
            signature=data["signature"],
            description=data["description"],
            parameters=data.get("parameters"),
            examples=data.get("examples", []),
            keywords=data.get("keywords", []),
        )


@dataclass
class RouteResult:
    """Result from routing a query"""
    query: str
    tool: str
    confidence: float
    latency_ms: float
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "query": self.query,
            "tool": self.tool,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "alternatives": self.alternatives,
            "metadata": self.metadata,
        }


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    output_dir: str

    # Training schedule
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Optimization
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Quantization
    quantization: Optional[Literal["4bit", "8bit"]] = "4bit"
    use_bf16: Optional[bool] = None  # Auto-detect if None

    # Logging & checkpointing
    logging_steps: int = 20
    save_steps: Optional[int] = None
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Reproducibility
    seed: int = 42

    # Advanced
    max_seq_length: int = 512
    push_to_hub: bool = False
    report_to: str = "none"

    def __post_init__(self):
        """Validate configuration"""
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")


@dataclass
class InferenceConfig:
    """Configuration for model inference"""
    device: str = "auto"
    batch_size: int = 1
    max_new_tokens: int = 30
    temperature: float = 0.0
    do_sample: bool = False
    return_alternatives: bool = True
    num_alternatives: int = 3
    enable_cache: bool = False
    cache_size: int = 1000


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation"""
    method: Literal["rule_based", "llm_based"] = "rule_based"

    # Rule-based settings
    num_variations: int = 50
    use_prefixes: bool = True
    use_suffixes: bool = True
    use_contexts: bool = True
    case_variations: bool = True

    # LLM-based settings
    num_samples: int = 1000
    llm_model: Optional[str] = "meta-llama/Llama-3.2-1B-Instruct"
    temperature: float = 0.7
    diversity_strategy: str = "paraphrase"  # "paraphrase", "expand", "persona"

    # Common settings
    domain_context: str = ""
    seed: int = 42

    def __post_init__(self):
        """Validate configuration"""
        if self.method not in ["rule_based", "llm_based"]:
            raise ValueError("method must be 'rule_based' or 'llm_based'")
        if self.num_variations < 1:
            raise ValueError("num_variations must be >= 1")
        if self.num_samples < 1:
            raise ValueError("num_samples must be >= 1")
