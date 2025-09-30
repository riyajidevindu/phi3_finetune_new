#!/usr/bin/env python3
"""Comprehensive Excel Dataset Analysis for Phi-3 Fine-tuning Project"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_excel_dataset(excel_path):
    """Comprehensive analysis of all sheets in the Excel dataset"""
    
    print("📊 COMPREHENSIVE EXCEL DATASET ANALYSIS FOR PHI-3 FINE-TUNING")
    print("=" * 80)
    
    # Read all sheets
    try:
        sheets_data = pd.read_excel(excel_path, sheet_name=None)
        print(f"📁 File: {excel_path}")
        print(f"📋 Total Sheets Found: {len(sheets_data)}")
        
        # List all sheets
        print(f"\n📚 SHEET NAMES:")
        for i, sheet_name in enumerate(sheets_data.keys(), 1):
            print(f"  {i}. '{sheet_name}' - {sheets_data[sheet_name].shape[0]} rows × {sheets_data[sheet_name].shape[1]} cols")
        
        return sheets_data
        
    except Exception as e:
        print(f"❌ Error reading Excel file: {str(e)}")
        return None

def analyze_individual_sheet(sheet_name, df, sheet_num):
    """Detailed analysis of individual sheet"""
    
    print(f"\n📋 SHEET {sheet_num}: '{sheet_name}'")
    print("-" * 60)
    
    print(f"📏 Shape: {df.shape} (rows × columns)")
    print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column analysis
    print(f"\n📊 COLUMNS ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        dtype = str(df[col].dtype)
        print(f"  {i:2d}. {col:<25} | {non_null:4d} non-null | {null_count:4d} null | {dtype}")
    
    # Check for key columns that indicate task types
    task_indicators = {
        'classification': ['Need Anonymization', 'Label', 'Class'],
        'extraction': ['Output', 'PII', 'Extracted'],
        'generation': ['Improved Prompt', 'Generated', 'Rewritten'],
        'original_text': ['Original', 'Input', 'Text', 'Prompt']
    }
    
    detected_tasks = []
    for task, indicators in task_indicators.items():
        if any(indicator in df.columns for indicator in indicators):
            detected_tasks.append(task)
    
    if detected_tasks:
        print(f"\n🎯 DETECTED TASK TYPES: {', '.join(detected_tasks)}")
    
    # Data distribution analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols[:3]:  # Analyze first 3 categorical columns
        if col in df.columns and df[col].notna().sum() > 0:
            unique_vals = df[col].value_counts().head(5)
            if len(unique_vals) <= 10:  # Only show if not too many unique values
                print(f"\n📈 '{col}' Distribution:")
                for val, count in unique_vals.items():
                    percentage = (count / len(df)) * 100
                    print(f"   • {str(val)[:50]}: {count} ({percentage:.1f}%)")
    
    # Sample data
    if len(df) > 0:
        print(f"\n🔍 SAMPLE ROWS (first 2):")
        for idx in range(min(2, len(df))):
            print(f"\n   Row {idx + 1}:")
            for col in df.columns[:4]:  # Show first 4 columns
                val = str(df.iloc[idx][col])[:100]
                if pd.isna(df.iloc[idx][col]):
                    val = "NaN"
                print(f"     {col}: {val}...")
    
    return df

def compare_sheets_for_fine_tuning(sheets_data):
    """Compare sheets and suggest fine-tuning strategy"""
    
    print(f"\n🚀 FINE-TUNING STRATEGY ANALYSIS")
    print("=" * 60)
    
    total_samples = sum(len(df) for df in sheets_data.values())
    print(f"📊 TOTAL SAMPLES ACROSS ALL SHEETS: {total_samples}")
    
    # Analyze task distribution
    task_sheets = {
        'classification': [],
        'extraction': [],
        'generation': [],
        'mixed': []
    }
    
    for sheet_name, df in sheets_data.items():
        has_classification = any(col in df.columns for col in ['Need Anonymization', 'Label', 'Class'])
        has_extraction = any(col in df.columns for col in ['Output', 'PII', 'Extracted'])
        has_generation = any(col in df.columns for col in ['Improved Prompt', 'Generated', 'Rewritten'])
        
        task_count = sum([has_classification, has_extraction, has_generation])
        
        if task_count > 1:
            task_sheets['mixed'].append((sheet_name, len(df)))
        elif has_classification:
            task_sheets['classification'].append((sheet_name, len(df)))
        elif has_extraction:
            task_sheets['extraction'].append((sheet_name, len(df)))
        elif has_generation:
            task_sheets['generation'].append((sheet_name, len(df)))
    
    print(f"\n🎯 TASK DISTRIBUTION:")
    for task_type, sheets in task_sheets.items():
        if sheets:
            total_samples_for_task = sum(count for _, count in sheets)
            print(f"   {task_type.title()}: {len(sheets)} sheets, {total_samples_for_task} samples")
            for sheet_name, count in sheets:
                print(f"     • {sheet_name}: {count} samples")
    
    # Training recommendations
    print(f"\n💡 TRAINING RECOMMENDATIONS:")
    
    if task_sheets['mixed']:
        print(f"   ✅ Multi-task training possible with {len(task_sheets['mixed'])} mixed sheets")
        print(f"   📚 Recommended approach: Single model trained on multiple objectives")
    
    if total_samples >= 1000:
        print(f"   ✅ Sufficient data ({total_samples} samples) for effective fine-tuning")
    elif total_samples >= 500:
        print(f"   ⚠️  Moderate data ({total_samples} samples) - consider data augmentation")
    else:
        print(f"   ❌ Limited data ({total_samples} samples) - may need more data or few-shot learning")
    
    # Memory optimization for 16GB GPU
    print(f"\n💾 16GB GPU OPTIMIZATION STRATEGY:")
    print(f"   • Model: Phi-3-mini-4k-instruct (3.8B parameters)")
    print(f"   • Quantization: QLoRA 4-bit (saves ~75% memory)")
    print(f"   • Batch size: 1-2 per GPU")
    print(f"   • Gradient accumulation: 8-16 steps")
    print(f"   • Max sequence length: 2048 tokens")
    print(f"   • Estimated VRAM usage: 12-14GB with QLoRA")
    
    return task_sheets

def create_training_data_splits(sheets_data):
    """Suggest how to split data for training"""
    
    print(f"\n📊 DATA SPLITTING RECOMMENDATIONS:")
    print("-" * 40)
    
    # Check if there are already train/val/test sheets
    sheet_names = [name.lower() for name in sheets_data.keys()]
    
    has_predefined_splits = any(
        split in ' '.join(sheet_names) 
        for split in ['train', 'val', 'test', 'validation']
    )
    
    if has_predefined_splits:
        print("✅ Pre-defined splits detected in sheet names")
        for sheet_name in sheets_data.keys():
            if any(split in sheet_name.lower() for split in ['train', 'val', 'test']):
                print(f"   • {sheet_name}: {len(sheets_data[sheet_name])} samples")
    else:
        print("📝 Recommended split strategy:")
        total_samples = sum(len(df) for df in sheets_data.values())
        train_size = int(total_samples * 0.8)
        val_size = int(total_samples * 0.15)
        test_size = total_samples - train_size - val_size
        
        print(f"   • Training: {train_size} samples (80%)")
        print(f"   • Validation: {val_size} samples (15%)")
        print(f"   • Test: {test_size} samples (5%)")
        print(f"   • Stratify by 'Need Anonymization' if available")

def save_analysis_summary(sheets_data, output_path):
    """Save analysis summary to file"""
    
    summary = {
        'total_sheets': len(sheets_data),
        'sheet_info': {},
        'total_samples': 0
    }
    
    for sheet_name, df in sheets_data.items():
        summary['sheet_info'][sheet_name] = {
            'rows': len(df),
            'columns': list(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        summary['total_samples'] += len(df)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Analysis summary saved to: {output_path}")

if __name__ == "__main__":
    excel_file = "/opt/projects/phi3_finetune_new/dataset/Prompt_sensitive_data.xlsx"
    
    # Main analysis
    sheets_data = analyze_excel_dataset(excel_file)
    
    if sheets_data:
        # Analyze each sheet individually
        for i, (sheet_name, df) in enumerate(sheets_data.items(), 1):
            analyze_individual_sheet(sheet_name, df, i)
        
        # Compare sheets and provide recommendations
        task_distribution = compare_sheets_for_fine_tuning(sheets_data)
        
        # Data splitting recommendations
        create_training_data_splits(sheets_data)
        
        # Save summary
        summary_path = "/opt/projects/phi3_finetune_new/analysis/dataset_analysis_summary.json"
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        save_analysis_summary(sheets_data, summary_path)
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📁 Next steps: Review the analysis and proceed with data preprocessing")
    else:
        print("❌ Failed to analyze dataset")