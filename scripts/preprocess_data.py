#!/usr/bin/env python3
"""
Data Preprocessing for Phi-3 Multi-Task PII Anonymization Fine-tuning
Converts Excel dataset to instruction-tuning format for single-model multi-task approach
"""

import pandas as pd
import json
import random
from pathlib import Path
from typing import Dict, List, Any
import re

class Phi3PIIDataProcessor:
    """Process PII dataset for Phi-3 multi-task fine-tuning"""
    
    def __init__(self, excel_path: str, output_dir: str):
        self.excel_path = Path(excel_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Instruction templates for variety
        self.instruction_templates = [
            "Analyze this prompt for PII and provide anonymization guidance: '{prompt}'",
            "Examine the following text for personally identifiable information and suggest anonymization: '{prompt}'",
            "Review this prompt for sensitive data and provide anonymization recommendations: '{prompt}'",
            "Check this text for PII and generate an anonymized version: '{prompt}'",
            "Assess this prompt for privacy concerns and provide anonymization: '{prompt}'"
        ]
        
        # System prompt for consistency
        self.system_prompt = """You are a privacy-focused AI assistant specialized in detecting personally identifiable information (PII) and providing anonymization guidance. 

Your task is to analyze text prompts and provide structured output with:
1. Whether the prompt needs anonymization (Yes/No)
2. Detected PII categories and values (if any)
3. Recommended anonymization technique (if needed)
4. An improved, anonymized version of the prompt (if needed)

Always respond with valid JSON format."""

    def load_excel_data(self) -> Dict[str, pd.DataFrame]:
        """Load all sheets from Excel file"""
        print(f"📊 Loading Excel data from {self.excel_path}")
        
        try:
            sheets_data = pd.read_excel(self.excel_path, sheet_name=None)
            print(f"✅ Loaded {len(sheets_data)} sheets")
            
            for sheet_name, df in sheets_data.items():
                print(f"   • {sheet_name}: {len(df)} samples")
            
            return sheets_data
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            raise

    def clean_json_string(self, json_str: str) -> str:
        """Clean and validate JSON string"""
        if pd.isna(json_str) or json_str.strip() == "":
            return "{}"
        
        try:
            # Parse and re-stringify to ensure valid JSON
            parsed = json.loads(json_str)
            return json.dumps(parsed, ensure_ascii=False)
        except:
            return "{}"

    def format_single_sample(self, row: pd.Series, sheet_name: str) -> Dict[str, Any]:
        """Format a single sample for instruction tuning"""
        
        # Get original prompt
        original_prompt = str(row['Original']).strip()
        need_anonymization = str(row['Need Anonymization']).strip()
        
        # Handle detections (PII output)
        detections = self.clean_json_string(row.get('Output', '{}'))
        
        # Handle anonymization technique
        technique = str(row.get('Anonymization Technique', '')).strip()
        if pd.isna(row.get('Anonymization Technique')) or technique == 'nan':
            technique = ""
        
        # Handle improved prompt
        improved_prompt = str(row.get('Improved Prompt', '')).strip()
        if pd.isna(row.get('Improved Prompt')) or improved_prompt == 'nan':
            improved_prompt = original_prompt  # Use original if no improvement
        
        # Create structured output
        if need_anonymization.lower() == 'yes':
            output_json = {
                "Need Anonymization": "Yes",
                "Detections": json.loads(detections) if detections != "{}" else {},
                "Anonymization Technique": technique,
                "Improved Prompt": improved_prompt
            }
        else:
            output_json = {
                "Need Anonymization": "No",
                "Detections": {},
                "Anonymization Technique": "",
                "Improved Prompt": original_prompt
            }
        
        # Choose random instruction template
        instruction = random.choice(self.instruction_templates).format(prompt=original_prompt)
        
        # Create ChatML format for Phi-3
        formatted_sample = {
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user", 
                    "content": instruction
                },
                {
                    "role": "assistant",
                    "content": json.dumps(output_json, ensure_ascii=False, indent=2)
                }
            ],
            "metadata": {
                "sheet_name": sheet_name,
                "has_pii": need_anonymization.lower() == 'yes',
                "pii_categories": list(json.loads(detections).keys()) if detections != "{}" else []
            }
        }
        
        return formatted_sample

    def process_all_sheets(self) -> List[Dict[str, Any]]:
        """Process all sheets and combine data"""
        
        sheets_data = self.load_excel_data()
        all_samples = []
        
        print(f"\n🔄 Processing samples...")
        
        for sheet_name, df in sheets_data.items():
            print(f"\n📋 Processing sheet: {sheet_name}")
            
            valid_samples = 0
            for idx, row in df.iterrows():
                try:
                    # Skip rows with missing critical data
                    if pd.isna(row['Original']) or pd.isna(row['Need Anonymization']):
                        continue
                    
                    formatted_sample = self.format_single_sample(row, sheet_name)
                    all_samples.append(formatted_sample)
                    valid_samples += 1
                    
                except Exception as e:
                    print(f"⚠️  Error processing row {idx}: {e}")
                    continue
            
            print(f"   ✅ Processed {valid_samples}/{len(df)} samples")
        
        print(f"\n📊 Total processed samples: {len(all_samples)}")
        return all_samples

    def create_train_val_test_splits(self, samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Create stratified train/validation/test splits"""
        
        print(f"\n🔀 Creating train/validation/test splits...")
        
        # Separate PII and non-PII samples for stratification
        pii_samples = [s for s in samples if s['metadata']['has_pii']]
        non_pii_samples = [s for s in samples if not s['metadata']['has_pii']]
        
        print(f"   • PII samples: {len(pii_samples)}")
        print(f"   • Non-PII samples: {len(non_pii_samples)}")
        
        # Shuffle samples
        random.shuffle(pii_samples)
        random.shuffle(non_pii_samples)
        
        def split_samples(samples_list, ratios=(0.8, 0.15, 0.05)):
            """Split samples according to ratios"""
            n = len(samples_list)
            train_n = int(n * ratios[0])
            val_n = int(n * ratios[1])
            
            return {
                'train': samples_list[:train_n],
                'validation': samples_list[train_n:train_n + val_n],
                'test': samples_list[train_n + val_n:]
            }
        
        # Split PII and non-PII samples separately
        pii_splits = split_samples(pii_samples)
        non_pii_splits = split_samples(non_pii_samples)
        
        # Combine splits
        splits = {
            'train': pii_splits['train'] + non_pii_splits['train'],
            'validation': pii_splits['validation'] + non_pii_splits['validation'],
            'test': pii_splits['test'] + non_pii_splits['test']
        }
        
        # Shuffle combined splits
        for split_name in splits:
            random.shuffle(splits[split_name])
        
        print(f"   ✅ Train: {len(splits['train'])} samples")
        print(f"   ✅ Validation: {len(splits['validation'])} samples")
        print(f"   ✅ Test: {len(splits['test'])} samples")
        
        return splits

    def save_splits(self, splits: Dict[str, List[Dict[str, Any]]]):
        """Save train/validation/test splits to JSONL files"""
        
        print(f"\n💾 Saving splits to {self.output_dir}...")
        
        for split_name, samples in splits.items():
            output_file = self.output_dir / f"{split_name}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"   ✅ Saved {split_name}: {len(samples)} samples → {output_file}")

    def create_dataset_info(self, splits: Dict[str, List[Dict[str, Any]]]):
        """Create dataset information file"""
        
        info = {
            "dataset_name": "phi3_pii_anonymization",
            "description": "Multi-task PII detection and anonymization dataset for Phi-3",
            "total_samples": sum(len(samples) for samples in splits.values()),
            "splits": {
                name: {
                    "count": len(samples),
                    "pii_samples": sum(1 for s in samples if s['metadata']['has_pii']),
                    "non_pii_samples": sum(1 for s in samples if not s['metadata']['has_pii'])
                }
                for name, samples in splits.items()
            },
            "task_format": "multi_task_instruction_following",
            "output_format": "structured_json",
            "fields": [
                "Need Anonymization",
                "Detections", 
                "Anonymization Technique",
                "Improved Prompt"
            ],
            "model_target": "microsoft/Phi-3-mini-4k-instruct",
            "training_method": "qlora_4bit"
        }
        
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Dataset info saved → {info_file}")

    def process_dataset(self):
        """Main processing pipeline"""
        
        print("🚀 STARTING PHI-3 PII DATASET PROCESSING")
        print("=" * 60)
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Process all samples
        samples = self.process_all_sheets()
        
        if not samples:
            print("❌ No valid samples found!")
            return
        
        # Create splits
        splits = self.create_train_val_test_splits(samples)
        
        # Save everything
        self.save_splits(splits)
        self.create_dataset_info(splits)
        
        print(f"\n🎉 PROCESSING COMPLETE!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Ready for Phi-3 fine-tuning!")

if __name__ == "__main__":
    # Configuration
    excel_path = "/opt/projects/phi3_finetune_new/dataset/Prompt_sensitive_data.xlsx"
    output_dir = "/opt/projects/phi3_finetune_new/data/processed"
    
    # Process dataset
    processor = Phi3PIIDataProcessor(excel_path, output_dir)
    processor.process_dataset()