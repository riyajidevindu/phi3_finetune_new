#!/usr/bin/env python3
"""
Custom Inference Script for Your Prompts
Test your own prompts with the trained model
"""

import sys
sys.path.append('/opt/projects/phi3_finetune_new')
from scripts.inference import Phi3PIIAnonymizer
import json

def custom_inference():
    """Interactive inference for your custom prompts"""
    
    print("🎯 PHI-3 PII ANONYMIZATION - CUSTOM TESTING")
    print("=" * 50)
    
    # Initialize model
    model_path = "/opt/projects/phi3_finetune_new/models/final/phi3-pii-anonymizer"
    print("🔄 Loading your trained model...")
    anonymizer = Phi3PIIAnonymizer(model_path)
    print("✅ Model loaded successfully!")
    
    print("\n💡 Enter your prompts to test PII detection and anonymization")
    print("   Type 'quit' to exit\n")
    
    while True:
        # Get user input
        user_prompt = input("🔍 Enter prompt to analyze: ").strip()
        
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not user_prompt:
            print("⚠️  Please enter a prompt")
            continue
        
        print(f"\n🔄 Analyzing: {user_prompt[:60]}...")
        print("-" * 40)
        
        try:
            # Analyze the prompt
            result = anonymizer.analyze_prompt(user_prompt)
            
            # Display results
            print("📋 ANALYSIS RESULT:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            # Summary
            needs_anon = result.get("Need Anonymization", "Unknown")
            detections = result.get("Detections", {})
            
            print(f"\n📊 SUMMARY:")
            print(f"   Needs Anonymization: {needs_anon}")
            print(f"   PII Types Detected: {len(detections)} types")
            if detections:
                for pii_type, values in detections.items():
                    print(f"     - {pii_type}: {len(values)} instances")
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"❌ Error analyzing prompt: {e}")
            print("Please try a different prompt.\n")

if __name__ == "__main__":
    custom_inference()