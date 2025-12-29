"""
Synthetic data generation for training

Supports:
1. Rule-based generation (like train.py lines 293-463)
2. LLM-based generation (enhancement)
"""

import random
from typing import List, Dict, Optional
from tqdm import tqdm

from funcroute.core.config import ToolDefinition, SyntheticDataConfig


class SyntheticDataGenerator:
    """
    Generate synthetic training data from tool definitions.

    Supports two methods:
    1. Rule-based: Pattern expansion with prefixes, suffixes, etc.
    2. LLM-based: Use language model to generate paraphrases
    """

    def __init__(
        self,
        method: str = "rule_based",
        llm_model: Optional[str] = None,
        temperature: float = 0.7,
        seed: int = 42,
    ):
        """
        Initialize generator.

        Args:
            method: "rule_based" or "llm_based"
            llm_model: HuggingFace model for LLM-based generation
            temperature: Temperature for LLM generation
            seed: Random seed for reproducibility
        """
        self.method = method
        self.llm_model = llm_model
        self.temperature = temperature
        self.seed = seed

        random.seed(seed)

        if method == "llm_based" and llm_model is None:
            self.llm_model = "meta-llama/Llama-3.2-1B-Instruct"

        # Rule-based variation templates (from train.py lines 403-414)
        self.prefixes = [
            "", "Please ", "Can you ", "Could you ", "Would you ", "Help me ",
            "I need to ", "I want to ", "I'd like to ", "Hey, ", "Hi, ", "Hello, ",
            "Urgent: ", "Quick question: ", "Excuse me, ", "Sorry, "
        ]

        self.suffixes = [
            "", "?", "!", " please", " thanks", " thank you", " now", " ASAP",
            " urgently", " right away", " if possible", " when you can",
            " pls", " plz", " thx", " quickly", " soon"
        ]

        self.contexts = [
            "", "Quick question - ", "I have a question - ", "Need help - ",
            "Can someone help? ", "This is urgent - ", "Important: "
        ]

        self.time_markers = [
            "", " today", " right now", " immediately", " this week", " soon"
        ]

    def generate(
        self,
        tools: List[ToolDefinition],
        num_variations: int = 50,
        num_samples: int = 1000,
        domain_context: str = "",
    ) -> List[Dict[str, str]]:
        """
        Generate synthetic training data.

        Args:
            tools: List of ToolDefinitions with examples
            num_variations: Number of variations per example (rule-based)
            num_samples: Total samples to generate (llm-based)
            domain_context: Domain context for generation

        Returns:
            List of {"query": str, "tool": str, "base_pattern": str}

        Example:
            >>> tools = [
            ...     ToolDefinition(
            ...         name="manage_order",
            ...         examples=["Where is my order?", "Track package"]
            ...     )
            ... ]
            >>> generator = SyntheticDataGenerator(method="rule_based")
            >>> data = generator.generate(tools, num_variations=50)
        """
        if self.method == "rule_based":
            return self._generate_rule_based(tools, num_variations, domain_context)
        elif self.method == "llm_based":
            return self._generate_llm_based(tools, num_samples, domain_context)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _generate_rule_based(
        self,
        tools: List[ToolDefinition],
        num_variations: int,
        domain_context: str,
    ) -> List[Dict[str, str]]:
        """
        Generate data using rule-based pattern expansion.

        Based on train.py lines 293-463.

        Args:
            tools: Tool definitions with examples
            num_variations: Number of variations per example
            domain_context: Domain context (unused in rule-based)

        Returns:
            List of samples
        """
        print("\n" + "=" * 80)
        print("RULE-BASED SYNTHETIC DATA GENERATION")
        print("=" * 80)
        print(f"Method: Rule-based pattern expansion")
        print(f"Variations per example: {num_variations}")
        print(f"Tools: {len(tools)}")
        print("=" * 80)

        pattern_groups = []

        # For each tool
        for tool in tqdm(tools, desc="Processing tools"):
            if not tool.examples:
                print(f"⚠️  Tool '{tool.name}' has no examples, skipping")
                continue

            # For each example in the tool
            for base_pattern in tool.examples:
                group = self._create_pattern_group(
                    base_pattern,
                    tool.name,
                    num_variations
                )
                pattern_groups.append(group)

        # Shuffle groups
        random.shuffle(pattern_groups)

        # Expand to individual samples
        samples = []
        for group in pattern_groups:
            for variation in group["variations"]:
                samples.append({
                    "query": variation,
                    "tool": group["tool"],
                    "base_pattern": group["base_pattern"],
                })

        print(f"\n✅ Generated {len(pattern_groups)} pattern groups")
        print(f"✅ Total samples: {len(samples):,}")

        return samples

    def _create_pattern_group(
        self,
        base_pattern: str,
        tool_name: str,
        num_variations: int,
    ) -> Dict:
        """
        Create variations of a base pattern.

        Uses 8 variation strategies (from train.py lines 431-448).

        Args:
            base_pattern: Base query pattern
            tool_name: Tool name
            num_variations: Number of variations to create

        Returns:
            Pattern group dict
        """
        variations = [base_pattern]  # Include original

        attempts = 0
        max_attempts = num_variations * 4

        while len(variations) < num_variations and attempts < max_attempts:
            attempts += 1

            # Choose variation strategy (1-8)
            strategy = random.randint(1, 8)

            if strategy == 1:
                # Prefix + suffix
                varied = f"{random.choice(self.prefixes)}{base_pattern.lower()}{random.choice(self.suffixes)}"

            elif strategy == 2:
                # Context prefix
                varied = f"{random.choice(self.contexts)}{base_pattern.lower()}"

            elif strategy == 3:
                # Time marker
                varied = f"{base_pattern.lower()}{random.choice(self.time_markers)}"

            elif strategy == 4:
                # Context + prefix + suffix
                varied = f"{random.choice(self.contexts)}{random.choice(self.prefixes)}{base_pattern.lower()}{random.choice(self.suffixes)}"

            elif strategy == 5:
                # Remove punctuation
                varied = base_pattern.lower().replace("?", "").replace("!", "")

            elif strategy == 6:
                # Uppercase
                varied = base_pattern.upper()

            elif strategy == 7:
                # Time + suffix
                varied = f"{base_pattern.lower()}{random.choice(self.time_markers)}{random.choice(self.suffixes)}"

            else:
                # Complex combination
                varied = f"{random.choice(self.contexts)}{random.choice(self.prefixes)}{base_pattern.lower()}{random.choice(self.time_markers)}{random.choice(self.suffixes)}"

            # Clean up and add if unique
            varied = varied.strip()
            if varied and varied not in variations:
                variations.append(varied)

        return {
            "base_pattern": base_pattern,
            "tool": tool_name,
            "variations": variations,
        }

    def _generate_llm_based(
        self,
        tools: List[ToolDefinition],
        num_samples: int,
        domain_context: str,
    ) -> List[Dict[str, str]]:
        """
        Generate data using LLM paraphrasing.

        Args:
            tools: Tool definitions with examples
            num_samples: Total number of samples to generate
            domain_context: Domain context for generation

        Returns:
            List of samples
        """
        print("\n" + "=" * 80)
        print("LLM-BASED SYNTHETIC DATA GENERATION")
        print("=" * 80)
        print(f"Method: LLM-based paraphrasing")
        print(f"Model: {self.llm_model}")
        print(f"Target samples: {num_samples}")
        print(f"Tools: {len(tools)}")
        print(f"Domain: {domain_context or 'General'}")
        print("=" * 80)

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch required for LLM-based generation. "
                "Install with: pip install transformers torch"
            )

        # Load model
        print(f"\nLoading model: {self.llm_model}...")
        tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("✅ Model loaded")

        # Generate samples
        samples = []
        samples_per_tool = num_samples // len(tools)

        for tool in tqdm(tools, desc="Generating for tools"):
            if not tool.examples:
                continue

            tool_samples = self._generate_for_tool_llm(
                tool,
                samples_per_tool,
                model,
                tokenizer,
                domain_context,
            )
            samples.extend(tool_samples)

        print(f"\n✅ Generated {len(samples):,} samples")

        return samples

    def _generate_for_tool_llm(
        self,
        tool: ToolDefinition,
        num_samples: int,
        model,
        tokenizer,
        domain_context: str,
    ) -> List[Dict[str, str]]:
        """
        Generate samples for a single tool using LLM.

        Args:
            tool: Tool definition
            num_samples: Number of samples to generate
            model: LLM model
            tokenizer: LLM tokenizer
            domain_context: Domain context

        Returns:
            List of samples for this tool
        """
        import torch

        samples = []
        base_examples = tool.examples[:3]  # Use first 3 examples

        # Calculate how many per example
        per_example = num_samples // len(base_examples)

        for base_pattern in base_examples:
            # Create prompt
            prompt = self._create_paraphrase_prompt(
                base_pattern,
                tool.description,
                domain_context,
                per_example,
            )

            # Generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=self.temperature,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract paraphrases
            paraphrases = self._extract_paraphrases(generated, base_pattern)

            # Add to samples
            for paraphrase in paraphrases[:per_example]:
                samples.append({
                    "query": paraphrase,
                    "tool": tool.name,
                    "base_pattern": base_pattern,
                })

        return samples

    def _create_paraphrase_prompt(
        self,
        example: str,
        description: str,
        domain_context: str,
        num_paraphrases: int,
    ) -> str:
        """Create prompt for paraphrase generation"""
        domain_text = f" in the {domain_context} domain" if domain_context else ""

        prompt = f"""Generate {num_paraphrases} different ways to ask the same question{domain_text}.

Original question: "{example}"
Context: {description}

Generate natural variations that mean the same thing. Make them diverse.

Paraphrases:
1."""

        return prompt

    def _extract_paraphrases(self, generated_text: str, base_pattern: str) -> List[str]:
        """
        Extract paraphrases from LLM output.

        Args:
            generated_text: Generated text from LLM
            base_pattern: Original pattern

        Returns:
            List of paraphrases
        """
        paraphrases = [base_pattern]  # Include original

        # Simple extraction: look for numbered lines
        lines = generated_text.split("\n")
        for line in lines:
            line = line.strip()
            # Match patterns like "1. query" or "- query"
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering
                clean = line.lstrip("0123456789.-) ").strip()
                if clean and clean not in paraphrases and len(clean) > 5:
                    paraphrases.append(clean)

        return paraphrases
