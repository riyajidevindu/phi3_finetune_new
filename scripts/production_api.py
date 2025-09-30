#!/usr/bin/env python3
"""
Production-Ready PII Anonymization API
For integrating into your applications
"""

import sys
sys.path.append('/opt/projects/phi3_finetune_new')
from scripts.inference import Phi3PIIAnonymizer
import json
from typing import Dict, List, Any
import logging

class PIIAnonymizationService:
    """Production service for PII detection and anonymization"""
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = "/opt/projects/phi3_finetune_new/models/final/phi3-pii-anonymizer"
        
        self.anonymizer = Phi3PIIAnonymizer(model_path)
        self.logger = logging.getLogger(__name__)
        
    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Process a single prompt for PII detection and anonymization
        
        Args:
            prompt: The text prompt to analyze
            
        Returns:
            Dictionary with your exact required format:
            {
                "Need Anonymization": "Yes/No",
                "Detections": {...},
                "Anonymization Technique": "...",
                "Improved Prompt": "..."
            }
        """
        try:
            result = self.anonymizer.analyze_prompt(prompt)
            
            # Add processing metadata
            result["processing_status"] = "success"
            result["original_prompt_length"] = len(prompt)
            result["pii_count"] = sum(len(v) if isinstance(v, list) else 1 
                                    for v in result.get("Detections", {}).values())
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing prompt: {e}")
            return {
                "Need Anonymization": "Error",
                "Detections": {},
                "Anonymization Technique": "",
                "Improved Prompt": prompt,
                "processing_status": "error",
                "error_message": str(e)
            }
    
    def batch_process(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple prompts"""
        results = []
        
        for i, prompt in enumerate(prompts):
            self.logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            result = self.process_prompt(prompt)
            result["batch_index"] = i
            results.append(result)
            
        return results
    
    def get_pii_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary statistics from batch processing"""
        
        total_prompts = len(results)
        needs_anonymization = sum(1 for r in results 
                                if r.get("Need Anonymization") == "Yes")
        
        all_pii_types = set()
        total_pii_instances = 0
        
        for result in results:
            detections = result.get("Detections", {})
            all_pii_types.update(detections.keys())
            total_pii_instances += sum(len(v) if isinstance(v, list) else 1 
                                     for v in detections.values())
        
        return {
            "total_prompts": total_prompts,
            "prompts_needing_anonymization": needs_anonymization,
            "anonymization_rate": needs_anonymization / total_prompts if total_prompts > 0 else 0,
            "unique_pii_types_detected": list(all_pii_types),
            "total_pii_instances": total_pii_instances,
            "average_pii_per_prompt": total_pii_instances / total_prompts if total_prompts > 0 else 0
        }

def demo_production_usage():
    """Demonstrate production usage"""
    
    print("🏭 PRODUCTION PII ANONYMIZATION SERVICE DEMO")
    print("=" * 50)
    
    # Initialize service
    service = PIIAnonymizationService()
    
    # Test prompts
    test_prompts = [
        "Hi, I'm Sarah Johnson, call me at 555-1234",
        "My email is user@company.com and I live in New York",
        "What is machine learning?",
        "I work at Google and my ID is ABC123",
        "Contact Dr. Smith at medical.center@hospital.org"
    ]
    
    print(f"📊 Processing {len(test_prompts)} test prompts...")
    
    # Batch processing
    results = service.batch_process(test_prompts)
    
    # Display results
    for i, result in enumerate(results):
        print(f"\n🔍 Prompt {i+1}: {test_prompts[i][:50]}...")
        print(f"   Needs Anonymization: {result['Need Anonymization']}")
        print(f"   PII Detected: {result['pii_count']} instances")
        if result.get("Detections"):
            print(f"   Types: {list(result['Detections'].keys())}")
    
    # Summary
    summary = service.get_pii_summary(results)
    print(f"\n📈 BATCH SUMMARY:")
    print(f"   Total Prompts: {summary['total_prompts']}")
    print(f"   Need Anonymization: {summary['prompts_needing_anonymization']}")
    print(f"   Anonymization Rate: {summary['anonymization_rate']:.1%}")
    print(f"   PII Types Found: {summary['unique_pii_types_detected']}")
    print(f"   Total PII Instances: {summary['total_pii_instances']}")

if __name__ == "__main__":
    demo_production_usage()