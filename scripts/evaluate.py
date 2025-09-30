#!/usr/bin/env python3
"""
Evaluation Script for Phi-3 PII Anonymization Model
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from scripts.inference import Phi3PIIAnonymizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phi3PIIEvaluator:
    """Evaluate the fine-tuned Phi-3 PII model"""
    
    def __init__(self, model_path: str, test_data_path: str):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.anonymizer = Phi3PIIAnonymizer(model_path)
        
    def load_test_data(self) -> List[Dict]:
        """Load test dataset"""
        test_data = []
        
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line.strip()))
        
        logger.info(f"📊 Loaded {len(test_data)} test samples")
        return test_data
    
    def evaluate_sample(self, sample: Dict) -> Dict[str, Any]:
        """Evaluate a single sample"""
        
        # Extract original prompt from the test sample
        messages = sample.get('messages', [])
        user_message = None
        expected_response = None
        
        for msg in messages:
            if msg['role'] == 'user':
                # Extract original prompt from user message
                content = msg['content']
                if "Analyze this prompt for PII" in content:
                    start = content.find("'") + 1
                    end = content.rfind("'")
                    user_message = content[start:end] if start > 0 and end > start else content
            elif msg['role'] == 'assistant':
                expected_response = msg['content']
        
        if not user_message:
            return {"error": "Could not extract original prompt"}
        
        # Get model prediction
        predicted = self.anonymizer.analyze_prompt(user_message)
        
        # Parse expected response
        try:
            expected = json.loads(expected_response)
        except:
            expected = {"error": "Could not parse expected response"}
        
        return {
            "original_prompt": user_message,
            "expected": expected,
            "predicted": predicted,
            "correct_anonymization_need": (
                expected.get("Need Anonymization", "").lower() == 
                predicted.get("Need Anonymization", "").lower()
            ),
            "has_detections": len(predicted.get("Detections", {})) > 0,
            "has_improved_prompt": bool(predicted.get("Improved Prompt", "").strip())
        }
    
    def run_evaluation(self, max_samples: int = 50) -> Dict[str, Any]:
        """Run evaluation on test set"""
        
        print("🔍 EVALUATING PHI-3 PII ANONYMIZATION MODEL")
        print("=" * 50)
        
        # Load test data
        test_data = self.load_test_data()
        
        # Limit samples for quick evaluation
        if len(test_data) > max_samples:
            test_data = test_data[:max_samples]
            logger.info(f"🎯 Limited to {max_samples} samples for evaluation")
        
        results = []
        correct_anonymization = 0
        total_with_pii = 0
        total_samples = len(test_data)
        
        print(f"📝 Evaluating {total_samples} samples...\n")
        
        for i, sample in enumerate(test_data):
            print(f"Processing {i+1}/{total_samples}...", end='\r')
            
            result = self.evaluate_sample(sample)
            results.append(result)
            
            # Count accuracy metrics
            if not result.get("error"):
                if result["correct_anonymization_need"]:
                    correct_anonymization += 1
                
                expected = result.get("expected", {})
                if expected.get("Need Anonymization", "").lower() == "yes":
                    total_with_pii += 1
        
        print(f"\n✅ Evaluation completed!")
        
        # Calculate metrics
        accuracy = correct_anonymization / total_samples if total_samples > 0 else 0
        
        # Summary statistics
        samples_with_detections = sum(1 for r in results if r.get("has_detections", False))
        samples_with_improved = sum(1 for r in results if r.get("has_improved_prompt", False))
        
        evaluation_summary = {
            "total_samples": total_samples,
            "anonymization_accuracy": accuracy,
            "correct_predictions": correct_anonymization,
            "samples_with_pii_detected": total_with_pii,
            "samples_with_detections": samples_with_detections,
            "samples_with_improved_prompts": samples_with_improved,
            "detection_rate": samples_with_detections / total_samples if total_samples > 0 else 0
        }
        
        return {
            "summary": evaluation_summary,
            "detailed_results": results
        }
    
    def print_evaluation_results(self, evaluation_results: Dict):
        """Print formatted evaluation results"""
        
        summary = evaluation_results["summary"]
        
        print("\n" + "="*50)
        print("📊 EVALUATION RESULTS")
        print("="*50)
        
        print(f"Total Samples Evaluated: {summary['total_samples']}")
        print(f"Anonymization Need Accuracy: {summary['anonymization_accuracy']:.2%}")
        print(f"Correct Predictions: {summary['correct_predictions']}/{summary['total_samples']}")
        print(f"Samples with PII Detected: {summary['samples_with_pii_detected']}")
        print(f"Model Detection Rate: {summary['detection_rate']:.2%}")
        print(f"Samples with Improved Prompts: {summary['samples_with_improved_prompts']}")
        
        # Show a few examples
        print(f"\n🔍 SAMPLE PREDICTIONS:")
        print("-" * 30)
        
        detailed = evaluation_results["detailed_results"]
        for i, result in enumerate(detailed[:3]):  # Show first 3 examples
            if result.get("error"):
                continue
                
            print(f"\nExample {i+1}:")
            print(f"Original: {result['original_prompt'][:80]}...")
            print(f"Predicted Need: {result['predicted'].get('Need Anonymization', 'N/A')}")
            print(f"Expected Need: {result['expected'].get('Need Anonymization', 'N/A')}")
            print(f"Correct: {'✅' if result['correct_anonymization_need'] else '❌'}")

def main():
    """Main evaluation entry point"""
    
    # Paths
    model_path = "/opt/projects/phi3_finetune_new/models/final/phi3-pii-anonymizer"
    test_data_path = "/opt/projects/phi3_finetune_new/data/processed/test.jsonl"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        print("   Please train the model first: python scripts/train_phi3.py")
        return
    
    # Check if test data exists
    if not Path(test_data_path).exists():
        print(f"❌ Test data not found at {test_data_path}")
        print("   Please run data preprocessing first: python scripts/preprocess_data.py")
        return
    
    # Run evaluation
    evaluator = Phi3PIIEvaluator(model_path, test_data_path)
    results = evaluator.run_evaluation(max_samples=20)  # Quick evaluation
    evaluator.print_evaluation_results(results)
    
    # Save results
    results_path = "/opt/projects/phi3_finetune_new/results/evaluation_results.json"
    Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {results_path}")

if __name__ == "__main__":
    main()