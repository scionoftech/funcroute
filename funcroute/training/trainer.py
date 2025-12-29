"""
Training orchestration for FunctionGemma fine-tuning

Based on train.py lines 843-1045
"""

import torch
import inspect
from typing import Optional, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from funcroute.core.config import TrainingConfig


class Trainer:
    """Orchestrate FunctionGemma fine-tuning with LoRA"""

    BASE_MODEL = "google/functiongemma-270m-it"

    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.

        Args:
            config: TrainingConfig with training parameters
        """
        self.config = config
        self.device = self._detect_device()
        self.model = None
        self.tokenizer = None

    def _detect_device(self) -> torch.device:
        """Detect available device (CUDA/CPU)"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _setup_quantization_config(self) -> BitsAndBytesConfig:
        """
        Setup quantization configuration.

        From train.py lines 861-866
        """
        # Detect BF16 support
        use_bf16 = self.config.use_bf16
        if use_bf16 is None:  # Auto-detect
            use_bf16 = (
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            )

        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

        if self.config.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            return None

    def setup_model(self):
        """
        Load base model with quantization.

        From train.py lines 868-886
        """
        print(f"Loading base model: {self.BASE_MODEL}")

        # Setup quantization
        quantization_config = self._setup_quantization_config()

        # Detect compute dtype
        use_bf16 = self.config.use_bf16
        if use_bf16 is None:
            use_bf16 = (
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            )
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        print("✅ Tokenizer loaded")

        # Load model
        print("Loading model (this may take 1-2 minutes)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=compute_dtype,
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✅ Model loaded: {total_params / 1e9:.2f}B parameters")

        return self.model, self.tokenizer

    def setup_lora(self):
        """
        Apply LoRA configuration to model.

        From train.py lines 889-927
        """
        print("Setting up LoRA...")

        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        print("✅ LoRA adapters applied")

        return self.model

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
    ):
        """
        Execute training.

        From train.py lines 929-1045

        Args:
            train_dataset: Training dataset (HF Dataset format)
            val_dataset: Optional validation dataset

        Returns:
            Trained model
        """
        print("\n" + "=" * 80)
        print("TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Output dir: {self.config.output_dir}")
        print("=" * 80)

        # Detect BF16 support
        use_bf16 = self.config.use_bf16
        if use_bf16 is None:
            use_bf16 = (
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            bf16=use_bf16,
            fp16=not use_bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps if self.config.save_steps else None,
            eval_strategy=self.config.eval_strategy if val_dataset else "no",
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end if val_dataset else False,
            metric_for_best_model=self.config.metric_for_best_model if val_dataset else None,
            save_total_limit=self.config.save_total_limit,
            push_to_hub=self.config.push_to_hub,
            report_to=self.config.report_to,
            seed=self.config.seed,
        )

        # Setup trainer with version compatibility (from train.py lines 997-1023)
        print("\nInitializing SFTTrainer...")

        sft_params = inspect.signature(SFTTrainer.__init__).parameters

        trainer_kwargs = {
            "model": self.model,
            "args": training_args,
            "train_dataset": train_dataset,
        }

        if val_dataset is not None:
            trainer_kwargs["eval_dataset"] = val_dataset

        # Version compatibility: processing_class vs tokenizer
        if "processing_class" in sft_params:
            trainer_kwargs["processing_class"] = self.tokenizer
            print("Using 'processing_class' parameter (trl >= 0.8.0)")
        elif "tokenizer" in sft_params:
            trainer_kwargs["tokenizer"] = self.tokenizer
            print("Using 'tokenizer' parameter (trl < 0.8.0)")

        # Optional parameters
        if "dataset_text_field" in sft_params:
            trainer_kwargs["dataset_text_field"] = "text"

        if "max_seq_length" in sft_params:
            trainer_kwargs["max_seq_length"] = self.config.max_seq_length

        if "packing" in sft_params:
            trainer_kwargs["packing"] = False

        trainer = SFTTrainer(**trainer_kwargs)

        print("✅ Trainer initialized\n")
        print("=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)

        # Train
        import time
        start_time = time.time()

        trainer.train()

        end_time = time.time()
        training_time = (end_time - start_time) / 60

        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Training time: {training_time:.1f} minutes")
        print("=" * 80)

        return self.model, trainer

    def save_model(self, model_dir: Optional[str] = None):
        """
        Save trained model and tokenizer.

        Args:
            model_dir: Directory to save to (uses config.output_dir if None)
        """
        if model_dir is None:
            model_dir = self.config.output_dir

        print(f"\nSaving model to: {model_dir}")

        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        print("✅ Model saved")

        return model_dir
