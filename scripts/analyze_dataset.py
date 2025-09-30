#!/usr/bin/env python3
"""Dataset Analysis Script for Phi-3 Fine-tuning Project"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

def analyze_csv_dataset(file_path):
    """Analyze the CSV dataset to understand its structure and content"""
    
    print("📊 DATASET ANALYSIS FOR PHI-3 FINE-TUNING")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Basic statistics
    print(f"📁 File: {file_path}")
    print(f"📏 Shape: {df.shape} (rows × columns)")
    print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n📋 COLUMNS:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # Data distribution
    print("\n🎯 CLASSIFICATION DISTRIBUTION:")
    need_anon_dist = df['Need Anonymization'].value_counts()
    print(need_anon_dist)
    print(f"Balance ratio: {need_anon_dist.min() / need_anon_dist.max():.2f}")
    
    # Missing values
    print("\n❌ MISSING VALUES:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values found!")
    else:
        print(missing[missing > 0])
    
    # Sample data for different tasks
    print("\n🔍 SAMPLE DATA FOR FINE-TUNING TASKS:")
    print("-" * 40)
    
    # Task 1: Classification
    print("1️⃣ PII DETECTION (Classification Task):")
    yes_sample = df[df['Need Anonymization'] == 'Yes'].iloc[0]
    print(f"   Input: {yes_sample['Original'][:100]}...")
    print(f"   Label: {yes_sample['Need Anonymization']}")
    
    # Task 2: PII Extraction
    print("\n2️⃣ PII EXTRACTION (Information Extraction Task):")
    print(f"   Input: {yes_sample['Original'][:100]}...")
    try:
        pii_data = json.loads(yes_sample['Output'])
        print(f"   Output: Found {len(pii_data)} PII categories")
        for key, values in list(pii_data.items())[:3]:  # Show first 3 categories
            print(f"     - {key}: {len(values)} items")
    except:
        print(f"   Output: {yes_sample['Output'][:100]}...")
    
    # Task 3: Prompt Improvement
    print("\n3️⃣ PROMPT ANONYMIZATION (Text Rewriting Task):")
    print(f"   Original: {yes_sample['Original'][:80]}...")
    print(f"   Improved: {yes_sample['Improved Prompt'][:80]}...")
    
    # Semantic similarity analysis
    print("\n📈 SEMANTIC SIMILARITY SCORES:")
    sem_scores = df['Semantic Similarity Score'].dropna()
    if len(sem_scores) > 0:
        print(f"   Mean: {sem_scores.mean():.4f}")
        print(f"   Std:  {sem_scores.std():.4f}")
        print(f"   Range: {sem_scores.min():.4f} - {sem_scores.max():.4f}")
    
    # Anonymization techniques
    print("\n🛡️ ANONYMIZATION TECHNIQUES:")
    techniques = df['Anonymization Technique'].value_counts().head(5)
    for tech, count in techniques.items():
        print(f"   • {tech}: {count} samples")
    
    return df

def check_for_excel_file(dataset_dir):
    """Check if there's an Excel file in the dataset directory"""
    dataset_path = Path(dataset_dir)
    excel_files = list(dataset_path.glob("*.xlsx")) + list(dataset_path.glob("*.xls"))
    
    if excel_files:
        print(f"\n📂 EXCEL FILES FOUND:")
        for file in excel_files:
            print(f"   • {file.name}")
        return excel_files
    else:
        print(f"\n📂 No Excel files found in {dataset_dir}")
        return []

def suggest_fine_tuning_approach(df):
    """Suggest fine-tuning approaches based on dataset analysis"""
    
    print("\n🚀 FINE-TUNING RECOMMENDATIONS FOR PHI-3-MINI:")
    print("=" * 50)
    
    total_samples = len(df)
    has_pii = len(df[df['Need Anonymization'] == 'Yes'])
    
    print(f"💪 Dataset Strengths:")
    print(f"   • {total_samples} samples - Good size for fine-tuning")
    print(f"   • {has_pii} PII samples - Sufficient for learning sensitive data patterns")
    print(f"   • Multi-task data - Can train on classification, extraction & rewriting")
    print(f"   • Balanced classes - Good for avoiding bias")
    
    print(f"\n🎯 Recommended Training Tasks:")
    print(f"   1. PII Detection (Classification): Yes/No prediction")
    print(f"   2. PII Extraction (JSON generation): Extract structured PII")
    print(f"   3. Prompt Anonymization (Text rewriting): Generate safe prompts")
    
    print(f"\n💾 Memory Optimization for 16GB GPU:")
    print(f"   • Use QLoRA (4-bit quantization) for Phi-3-mini")
    print(f"   • Batch size: 1-2 (depending on sequence length)")
    print(f"   • Gradient accumulation: 4-8 steps")
    print(f"   • Max sequence length: 2048 tokens (fits Phi-3-mini context)")

if __name__ == "__main__":
    dataset_dir = "/opt/projects/phi3_finetune_new/dataset"
    csv_file = f"{dataset_dir}/Prompt_sensitive_data.csv"
    
    # Check for Excel files first
    excel_files = check_for_excel_file(dataset_dir)
    
    # Analyze CSV file
    df = analyze_csv_dataset(csv_file)
    
    # Provide recommendations
    suggest_fine_tuning_approach(df)
    
    if not excel_files:
        print(f"\n⚠️  RECOMMENDATION:")
        print(f"   If the original dataset has multiple sheets/tabs,")
        print(f"   please provide the .xlsx file for complete analysis!")