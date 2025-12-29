"""
Main FuncRoute class for intelligent function/tool routing

Based on train.py with production enhancements
"""

import re
import time
import torch
from typing import List, Dict, Optional, Union, Any
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import Dataset
from peft import PeftModel

from funcroute.core.config import (
    TrainingConfig,
    InferenceConfig,
    RouteResult,
    ToolDefinition,
)
from funcroute.data.formatter import DataFormatter
from funcroute.data.loader import DataLoader
from funcroute.training.trainer import Trainer


class FuncRoute:
    """
    Main router class for intelligent function/tool routing.

    Example:
        >>> # Train router
        >>> router = FuncRoute()
        >>> router.train(
        ...     train_data="train.jsonl",
        ...     config=TrainingConfig(output_dir="./my_router")
        ... )
        >>>
        >>> # Use router
        >>> result = router.route("Where is my order?")
        >>> print(result.tool)  # "manage_order"
    """

    def __init__(self, device: str = "auto"):
        """
        Initialize FuncRoute.

        Args:
            device: Device to use ("auto", "cuda", "cpu")
        """
        self.device = self._detect_device(device)
        self.model = None
        self.tokenizer = None
        self.tools = None
        self.tool_definitions_block = None
        self.formatter = DataFormatter()
        self.loader = DataLoader()

    def _detect_device(self, device: str) -> torch.device:
        """Detect and set device"""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def train(
        self,
        train_data: Union[str, List[Dict], Any],
        val_data: Optional[Union[str, List[Dict], Any]] = None,
        tools: Optional[List[ToolDefinition]] = None,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Train the routing model.

        Args:
            train_data: Training data (filepath, list of dicts, DataFrame, etc.)
            val_data: Optional validation data
            tools: Optional tool definitions (auto-extracted if not provided)
            config: Training configuration

        Example:
            >>> train_data = [
            ...     {"query": "Where is my order?", "tool": "manage_order"},
            ...     {"query": "Show me laptops", "tool": "search_products"},
            ... ]
            >>> router = FuncRoute()
            >>> router.train(
            ...     train_data=train_data,
            ...     config=TrainingConfig(output_dir="./router")
            ... )
        """
        if config is None:
            config = TrainingConfig(output_dir="./funcroute_model")

        print("\n" + "=" * 80)
        print("FUNCROUTE TRAINING")
        print("=" * 80)

        # Step 1: Load training data
        print("\n[1/6] Loading training data...")
        train_samples = self.loader.load(train_data)
        print(f"✅ Loaded {len(train_samples)} training samples")

        if val_data is not None:
            val_samples = self.loader.load(val_data)
            print(f"✅ Loaded {len(val_samples)} validation samples")
        else:
            val_samples = None

        # Step 2: Extract/use tool definitions
        print("\n[2/6] Preparing tool definitions...")
        if tools is None:
            # Auto-extract tools from data
            tool_names = self.formatter.extract_tools_from_data(train_samples)
            print(f"✅ Extracted {len(tool_names)} unique tools: {tool_names}")

            # Create simple tool definitions
            tools = self.formatter.create_tool_definitions_from_names(tool_names)
            print("✅ Created tool definitions")
        else:
            print(f"✅ Using {len(tools)} provided tool definitions")

        self.tools = tools

        # Step 3: Format data for FunctionGemma
        print("\n[3/6] Formatting data for FunctionGemma...")
        train_formatted = self.formatter.format_for_training(train_samples, tools)
        print(f"✅ Formatted {len(train_formatted)} training samples")

        # Validate format
        if not self.formatter.validate_format(train_formatted[0]["text"]):
            raise ValueError("FunctionGemma format validation failed!")

        if val_samples:
            val_formatted = self.formatter.format_for_training(val_samples, tools)
            print(f"✅ Formatted {len(val_formatted)} validation samples")
        else:
            val_formatted = None

        # Convert to HF Dataset
        train_dataset = Dataset.from_list(train_formatted)
        val_dataset = Dataset.from_list(val_formatted) if val_formatted else None

        # Step 4: Setup model and LoRA
        print("\n[4/6] Setting up model...")
        trainer = Trainer(config)
        trainer.setup_model()
        trainer.setup_lora()

        # Step 5: Train
        print("\n[5/6] Training...")
        model, trainer_obj = trainer.train(train_dataset, val_dataset)

        # Step 6: Save
        print("\n[6/6] Saving model...")
        model_dir = trainer.save_model()

        # Save tool definitions (CRITICAL for loading model later!)
        import json
        tool_defs_path = Path(model_dir) / "tool_definitions.json"
        with open(tool_defs_path, "w") as f:
            json.dump([t.to_dict() for t in tools], f, indent=2)
        print(f"✅ Tool definitions saved to: {tool_defs_path}")

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE")
        print("=" * 80)
        print(f"Model saved to: {model_dir}")
        print(f"\nTo use the model:")
        print(f'  router = FuncRoute.load("{model_dir}")')
        print(f'  result = router.route("your query")')
        print("=" * 80)

        # Load the trained model for immediate use
        self.model = model
        self.tokenizer = trainer.tokenizer
        self.tool_definitions_block = self.formatter._build_tool_definitions_block(tools)

        return model_dir

    @classmethod
    def load(cls, model_path: str, device: str = "auto") -> "FuncRoute":
        """
        Load a trained FuncRoute model.

        Args:
            model_path: Path to saved model directory
            device: Device to use ("auto", "cuda", "cpu")

        Returns:
            Loaded FuncRoute instance

        Example:
            >>> router = FuncRoute.load("./my_router")
            >>> result = router.route("Track my package")
        """
        instance = cls(device=device)

        print(f"Loading FuncRoute model from: {model_path}")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        instance.tokenizer.pad_token = instance.tokenizer.eos_token
        instance.tokenizer.padding_side = "right"

        # Load model (LoRA adapters already merged in saved model)
        instance.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="auto",
        )

        # Try to load tool definitions if available
        tool_def_path = model_path / "tool_definitions.json"
        if tool_def_path.exists():
            import json
            with open(tool_def_path) as f:
                tools_data = json.load(f)
            instance.tools = [ToolDefinition.from_dict(t) for t in tools_data]
            instance.tool_definitions_block = instance.formatter._build_tool_definitions_block(
                instance.tools
            )

        print("✅ Model loaded successfully")

        return instance

    @classmethod
    def from_pretrained(cls, model_name: str, device: str = "auto") -> "FuncRoute":
        """
        Load a pre-trained model from Hugging Face Hub.

        Args:
            model_name: Model name on HF Hub (e.g., "scionoftech/functiongemma-e-commerce-tool-calling")
            device: Device to use

        Returns:
            Loaded FuncRoute instance

        Example:
            >>> router = FuncRoute.from_pretrained("scionoftech/functiongemma-e-commerce-tool-calling")
            >>> result = router.route("Where is my order?")
        """
        instance = cls(device=device)

        print(f"Loading pre-trained model: {model_name}")

        # Load tokenizer
        instance.tokenizer = AutoTokenizer.from_pretrained(model_name)
        instance.tokenizer.pad_token = instance.tokenizer.eos_token
        instance.tokenizer.padding_side = "right"

        # Load model
        instance.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )

        print("✅ Pre-trained model loaded successfully")

        return instance

    def route(self, query: str, config: Optional[InferenceConfig] = None) -> RouteResult:
        """
        Route a query to the best tool.

        Based on train.py lines 1181-1234

        Args:
            query: User query to route
            config: Optional inference configuration

        Returns:
            RouteResult with selected tool and metadata

        Example:
            >>> result = router.route("Where is my package?")
            >>> print(result.tool)  # "manage_order"
            >>> print(result.confidence)  # 0.98
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call train() or load() first.")

        if config is None:
            config = InferenceConfig()

        # Build prompt with tool definitions
        if self.tool_definitions_block is None:
            # Fallback: try to extract from model config
            raise ValueError(
                "Tool definitions not available. "
                "Model may not have been trained with FuncRoute."
            )

        prompt = f"""<start_of_turn>user
{self.tool_definitions_block}

User query: {query}<end_of_turn>
<start_of_turn>model
"""

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Decode
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Extract function call (from train.py line 1230)
        match = re.search(
            r"<function_call>\s*([a-zA-Z_]+)\s*</function_call>",
            generated_text
        )

        if match:
            tool_name = match.group(1).strip()
        else:
            tool_name = "unknown"

        # Build result
        result = RouteResult(
            query=query,
            tool=tool_name,
            confidence=1.0,  # TODO: Calculate actual confidence
            latency_ms=latency_ms,
            alternatives=[],  # TODO: Get alternatives
            metadata={"generated_text": generated_text}
        )

        return result

    def route_batch(
        self,
        queries: List[str],
        config: Optional[InferenceConfig] = None
    ) -> List[RouteResult]:
        """
        Route multiple queries (enhancement over train.py).

        Args:
            queries: List of queries to route
            config: Optional inference configuration

        Returns:
            List of RouteResults
        """
        # Simple implementation: route one at a time
        # TODO: Optimize with actual batching
        return [self.route(query, config) for query in queries]

    def save(self, output_dir: str):
        """
        Save the model.

        Args:
            output_dir: Directory to save to
        """
        if self.model is None:
            raise ValueError("No model to save")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving model to: {output_dir}")

        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save tool definitions
        if self.tools:
            import json
            tool_defs_path = output_path / "tool_definitions.json"
            with open(tool_defs_path, "w") as f:
                json.dump([t.to_dict() for t in self.tools], f, indent=2)

        print("✅ Model saved")
