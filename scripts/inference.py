#!/usr/bin/env python3
"""
Phi-3 PII Anonymization Inference Script
Exactly matches your required output format
"""

import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phi3PIIAnonymizer:
    """Phi-3 PII Detection and Anonymization Inference"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        # System prompt (same as training)
        self.system_prompt = """You are a privacy-focused AI assistant specialized in detecting personally identifiable information (PII) and providing anonymization guidance. 

Your task is to analyze text prompts and provide structured output with:
1. Whether the prompt needs anonymization (Yes/No)
2. Detected PII categories and values (if any)
3. Recommended anonymization technique (if needed)
4. An improved, anonymized version of the prompt (if needed)

Always respond with valid JSON format."""
        
    def _load_tokenizer(self):
        """Load tokenizer"""
        logger.info(f"📝 Loading tokenizer from {self.model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"  # For inference
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    
    def _load_model(self):
        """Load fine-tuned model"""
        logger.info(f"🤖 Loading model from {self.model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  # Fix for cache compatibility
            use_cache=False  # Disable caching to avoid compatibility issues
        )
        
        model.eval()
        return model
    
    def _create_messages(self, original_prompt: str) -> list:
        """Create message format for inference"""
        return [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": f"Analyze this prompt for PII and provide anonymization guidance: '{original_prompt}'"
            }
        ]
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate model response"""
        try:
            # Find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response_text[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["Need Anonymization", "Detections", "Anonymization Technique", "Improved Prompt"]
            for field in required_fields:
                if field not in result:
                    result[field] = "" if field != "Detections" else {}
            
            # Ensure proper types
            if not isinstance(result["Detections"], dict):
                result["Detections"] = {}
                
            return result
            
        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")
            # Return default structure
            return {
                "Need Anonymization": "No",
                "Detections": {},
                "Anonymization Technique": "",
                "Improved Prompt": original_prompt
            }
    
    def analyze_prompt(self, original_prompt: str, max_length: int = 2048) -> Dict[str, Any]:
        """
        Analyze a prompt for PII and return anonymization guidance
        
        Args:
            original_prompt: The original text prompt to analyze
            max_length: Maximum generation length
            
        Returns:
            Dictionary with your exact required format:
            {
                "Need Anonymization": "Yes/No",
                "Detections": {...} or {},
                "Anonymization Technique": "technique" or "",
                "Improved Prompt": "anonymized text" or "original text"
            }
        """
        
        # Create messages
        messages = self._create_messages(original_prompt)
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024  # Leave room for generation
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.1,  # Low temperature for consistent output
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False  # Disable caching for compatibility
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Parse and return structured result
        return self._parse_response(response)
    
    def batch_analyze(self, prompts: list, max_length: int = 2048) -> list:
        """Analyze multiple prompts"""
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.analyze_prompt(prompt, max_length)
            results.append({
                "original_prompt": prompt,
                **result
            })
        
        return results

def demo_inference(model_path: str):
    """Demo the inference with sample prompts"""
    
    print("🚀 PHI-3 PII ANONYMIZATION DEMO")
    print("=" * 50)
    
    # Initialize anonymizer
    anonymizer = Phi3PIIAnonymizer(model_path)
    
    # Test prompts
    test_prompts = [
        # Prompt with PII
        "My name is John Doe and my SSN is 123-45-6789. I work at Microsoft and live at 123 Main St, Seattle, WA.",
        
        # Prompt without PII
        "Can you explain how machine learning works in simple terms?",
        
        # Prompt with mixed content
        "I'm researching AI ethics. My email is researcher@university.edu and I'm affiliated with Stanford University."
    ]
    
    print(f"Testing {len(test_prompts)} sample prompts...\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"🔍 Test {i}: {prompt[:60]}...")
        print("-" * 40)
        
        result = anonymizer.analyze_prompt(prompt)
        
        print("📋 RESULT:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("\n" + "="*50 + "\n")

def main():
    """Main inference entry point"""
    
    # Model path (update this to your trained model)
    model_path = "/opt/projects/phi3_finetune_new/models/final/phi3-pii-anonymizer"
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("   Please train the model first: python scripts/train_phi3.py")
        return
    
    # Run demo
    demo_inference(model_path)

if __name__ == "__main__":
    main()