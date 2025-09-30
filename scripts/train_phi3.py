#!/usr/bin/env python3
"""
Phi-3 PII Anonymization Fine-tuning Script
Optimized for 16GB GPU with QLoRA 4-bit quantization
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from dotenv import load_dotenv

import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phi3PIITrainer:
    """Phi-3 PII Anonymization Multi-Task Trainer"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.setup_environment()
        
    def load_config(self) -> Dict[str, Any]:
        """Load training configuration"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✅ Loaded configuration from {self.config_path}")
        return config
    
    def setup_environment(self):
        """Setup training environment"""
        # Set CUDA device
        if "cuda_visible_devices" in self.config:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.config["cuda_visible_devices"])
        
        # Memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Create output directories
        for dir_key in ["output_dir", "final_model_dir", "logging_dir"]:
            if dir_key in self.config:
                Path(self.config[dir_key]).mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Environment setup complete")
    
    def setup_wandb(self):
        """Initialize Weights & Biases logging"""
        
        # Check if WANDB_API_KEY is set
        wandb_api_key = os.getenv('WANDB_API_KEY')
        if not wandb_api_key:
            logger.warning("⚠️  WANDB_API_KEY not found in environment variables")
            logger.warning("   Please set it in your .env file or disable wandb logging")
            # Set wandb to offline mode if no API key
            os.environ['WANDB_MODE'] = 'offline'
        
        # Get project and entity from environment or use defaults
        wandb_project = os.getenv('WANDB_PROJECT', 'phi3-pii-anonymization')
        wandb_entity = os.getenv('WANDB_ENTITY', None)
        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"phi3-pii-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                **self.config,
                "model_name": "microsoft/Phi-3-mini-4k-instruct",
                "method": "QLoRA 4-bit",
                "task": "PII Detection & Anonymization",
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
            }
        )
        
        if os.getenv('WANDB_MODE') == 'offline':
            logger.info(f"🔄 W&B initialized in OFFLINE mode: {wandb.run.name}")
        else:
            logger.info(f"🔄 W&B initialized: {wandb.run.name}")
    
    def load_tokenizer(self):
        """Load and configure tokenizer"""
        logger.info(f"📝 Loading tokenizer: {self.config['model_name']}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_name"],
            cache_dir=self.config.get("model_cache_dir"),
            trust_remote_code=True,
            padding_side="right"  # Important for training
        )
        
        # Add special tokens if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        logger.info(f"✅ Tokenizer loaded. Vocab size: {len(tokenizer)}")
        return tokenizer
    
    def load_model(self):
        """Load and configure model with QLoRA"""
        logger.info(f"🤖 Loading model: {self.config['model_name']}")
        
        # BitsAndBytes configuration for 4-bit quantization
        qlora_config = self.config.get("qlora_config", {})
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=getattr(torch, qlora_config.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_quant_type=qlora_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=qlora_config.get("bnb_4bit_use_double_quant", False)
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            quantization_config=bnb_config,
            cache_dir=self.config.get("model_cache_dir"),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration
        lora_config = self.config.get("lora_config", {})
        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.1),
            bias=lora_config.get("bias", "none"),
            task_type=lora_config.get("task_type", "CAUSAL_LM"),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"])
        )
        
        # Apply LoRA
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        logger.info("✅ Model loaded with QLoRA configuration")
        return model
    
    def load_datasets(self) -> Dict[str, Dataset]:
        """Load and prepare datasets"""
        logger.info("📊 Loading datasets...")
        
        datasets = {}
        
        # Load train dataset
        if "train_file" in self.config and Path(self.config["train_file"]).exists():
            datasets["train"] = self.load_jsonl_dataset(self.config["train_file"])
            logger.info(f"✅ Train dataset: {len(datasets['train'])} samples")
        
        # Load validation dataset
        if "validation_file" in self.config and Path(self.config["validation_file"]).exists():
            datasets["validation"] = self.load_jsonl_dataset(self.config["validation_file"])
            logger.info(f"✅ Validation dataset: {len(datasets['validation'])} samples")
        
        # Load test dataset
        if "test_file" in self.config and Path(self.config["test_file"]).exists():
            datasets["test"] = self.load_jsonl_dataset(self.config["test_file"])
            logger.info(f"✅ Test dataset: {len(datasets['test'])} samples")
        
        return datasets
    
    def load_jsonl_dataset(self, file_path: str) -> Dataset:
        """Load dataset from JSONL file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        
        return Dataset.from_list(data)
    
    def preprocess_function(self, examples, tokenizer):
        """Preprocess examples for training"""
        
        # Apply chat template to format messages
        texts = []
        for messages in examples["messages"]:
            # Convert to the format expected by the tokenizer
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        
        # Tokenize
        model_inputs = tokenizer(
            texts,
            max_length=self.config.get("max_seq_length", 2048),
            truncation=True,
            padding=False,  # We'll pad dynamically
            return_tensors=None
        )
        
        # For causal language modeling, labels are the same as input_ids
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments"""
        
        args = TrainingArguments(
            output_dir=self.config["output_dir"],
            
            # Training parameters
            num_train_epochs=self.config.get("num_train_epochs", 3),
            per_device_train_batch_size=self.config.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=self.config.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 8),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
            
            # Optimization
            learning_rate=self.config.get("learning_rate", 2e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            warmup_ratio=self.config.get("warmup_ratio", 0.1),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),
            optim=self.config.get("optim", "paged_adamw_8bit"),
            lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine"),
            
            # Evaluation (updated parameter names)
            eval_strategy=self.config.get("eval_strategy", "steps"),  # Use correct parameter name
            eval_steps=self.config.get("eval_steps", 100),
            
            # Saving
            save_strategy=self.config.get("save_strategy", "steps"),
            save_steps=self.config.get("save_steps", 100),
            save_total_limit=self.config.get("save_total_limit", 3),
            load_best_model_at_end=self.config.get("load_best_model_at_end", True),
            metric_for_best_model=self.config.get("metric_for_best_model", "eval_loss"),
            greater_is_better=self.config.get("greater_is_better", False),  # Added for clarity
            
            # Logging
            logging_dir=self.config.get("logging_dir"),
            logging_strategy=self.config.get("logging_strategy", "steps"),
            logging_steps=self.config.get("logging_steps", 10),
            report_to=self.config.get("report_to", ["tensorboard"]),
            
            # Hardware optimization
            fp16=self.config.get("fp16", False),
            bf16=self.config.get("bf16", True),
            tf32=self.config.get("tf32", True),
            dataloader_num_workers=self.config.get("dataloader_num_workers", 4),
            dataloader_pin_memory=self.config.get("dataloader_pin_memory", True),
            group_by_length=self.config.get("group_by_length", True),
            remove_unused_columns=self.config.get("remove_unused_columns", False),
        )
        
        return args
    
    def train(self):
        """Main training function"""
        
        logger.info("🚀 Starting Phi-3 PII Anonymization Training")
        logger.info("=" * 60)
        
        # Initialize wandb with environment variables
        self.setup_wandb()
        
        # Load components
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        datasets = self.load_datasets()
        
        # Preprocess datasets
        logger.info("🔄 Preprocessing datasets...")
        
        def preprocess_with_tokenizer(examples):
            return self.preprocess_function(examples, tokenizer)
        
        if "train" in datasets:
            datasets["train"] = datasets["train"].map(
                preprocess_with_tokenizer,
                batched=True,
                remove_columns=datasets["train"].column_names
            )
        
        if "validation" in datasets:
            datasets["validation"] = datasets["validation"].map(
                preprocess_with_tokenizer,
                batched=True,
                remove_columns=datasets["validation"].column_names
            )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal language modeling
            pad_to_multiple_of=8  # For efficiency
        )
        
        # Training arguments
        training_args = self.setup_training_arguments()
        
        # Setup callbacks
        callbacks = []
        
        # Add early stopping if enabled
        if self.config.get("early_stopping", {}).get("enabled", False):
            early_stopping_config = self.config["early_stopping"]
            early_stopping = CustomEarlyStoppingCallback(
                early_stopping_patience=early_stopping_config.get("patience", 3),
                early_stopping_threshold=early_stopping_config.get("min_delta", 0.001)
            )
            callbacks.append(early_stopping)
            logger.info(f"🎯 Early stopping enabled: patience={early_stopping_config.get('patience', 3)}, min_delta={early_stopping_config.get('min_delta', 0.001)}")
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=datasets.get("train"),
            eval_dataset=datasets.get("validation"),
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=callbacks
        )
        
        # Check GPU memory before training
        if torch.cuda.is_available():
            logger.info(f"🔥 GPU Memory before training:")
            logger.info(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # Start training
        logger.info("🏋️ Starting training...")
        logger.info(f"📊 Training for up to {self.config['num_train_epochs']} epochs")
        if self.config.get("early_stopping", {}).get("enabled", False):
            logger.info("🎯 Early stopping is enabled - training may stop before max epochs")
        
        # Train the model
        train_result = trainer.train()
        
        # Log training summary
        logger.info("📈 Training Summary:")
        logger.info(f"   Total steps: {train_result.global_step}")
        logger.info(f"   Training loss: {train_result.training_loss:.6f}")
        
        # Final evaluation
        if "validation" in datasets:
            logger.info("🔍 Running final evaluation...")
            eval_results = trainer.evaluate()
            logger.info("📊 Final Evaluation Results:")
            for key, value in eval_results.items():
                if isinstance(value, (int, float)):
                    logger.info(f"   {key}: {value:.6f}")
        
        # Save final model
        logger.info("💾 Saving final model...")
        final_model_path = Path(self.config["final_model_dir"]) / "phi3-pii-anonymizer"
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        # Save training metrics
        training_metrics = {
            "final_training_loss": train_result.training_loss,
            "total_steps": train_result.global_step,
            "epochs_completed": train_result.global_step / len(trainer.get_train_dataloader()),
        }
        
        if "validation" in datasets:
            training_metrics.update(eval_results)
            
        metrics_file = final_model_path / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        
        # Save training config
        with open(final_model_path / "training_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"✅ Training complete! Model saved to {final_model_path}")
        logger.info(f"📋 Training metrics saved to {metrics_file}")
        
        # Final GPU memory check
        if torch.cuda.is_available():
            logger.info(f"🔥 GPU Memory after training:")
            logger.info(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        return final_model_path

def main():
    """Main training entry point"""
    
    # Configuration
    config_path = "/opt/projects/phi3_finetune_new/configs/training_config.yaml"
    
    # Create trainer and start training
    trainer = Phi3PIITrainer(config_path)
    final_model_path = trainer.train()
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"📁 Final model saved to: {final_model_path}")
    print(f"🚀 Ready for inference!")

if __name__ == "__main__":
    main()