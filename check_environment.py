#!/usr/bin/env python3
"""
Environment Status Check for Phi-3 Fine-tuning Project
"""

import sys
import os
import subprocess
from pathlib import Path

def check_environment():
    """Check the status of the Phi-3 fine-tuning environment"""
    
    print("🔍 PHI-3 FINE-TUNING ENVIRONMENT STATUS")
    print("=" * 60)
    
    # Check project structure
    project_root = Path("/opt/projects/phi3_finetune_new")
    required_dirs = [
        "dataset", "data/processed", "models/checkpoints", "models/final", 
        "models/cache", "scripts", "notebooks", "configs", "logs", "results", 
        "tests", "src/phi3_finetune", "phi3_env"
    ]
    
    print("📁 PROJECT STRUCTURE:")
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        status = "✅" if full_path.exists() else "❌"
        print(f"   {status} {dir_path}")
    
    # Check virtual environment
    venv_path = project_root / "phi3_env"
    venv_python = venv_path / "bin" / "python"
    
    print(f"\n🐍 VIRTUAL ENVIRONMENT:")
    print(f"   ✅ Location: {venv_path}")
    print(f"   ✅ Python: {venv_python}")
    
    # Check key files
    key_files = [
        "requirements.txt", "pyproject.toml", "setup_environment.sh",
        "activate_env.sh", "README.md", ".env.example"
    ]
    
    print(f"\n📄 KEY FILES:")
    for file_name in key_files:
        file_path = project_root / file_name
        status = "✅" if file_path.exists() else "❌"
        print(f"   {status} {file_name}")
    
    # Check dataset
    dataset_path = project_root / "dataset" / "Prompt_sensitive_data.xlsx"
    print(f"\n📊 DATASET:")
    if dataset_path.exists():
        print(f"   ✅ Excel dataset found: {dataset_path}")
        try:
            import pandas as pd
            excel_data = pd.read_excel(dataset_path, sheet_name=None)
            total_samples = sum(len(df) for df in excel_data.values())
            print(f"   ✅ Total samples: {total_samples}")
            print(f"   ✅ Sheets: {len(excel_data)}")
        except Exception as e:
            print(f"   ⚠️  Could not analyze dataset: {e}")
    else:
        print(f"   ❌ Dataset not found: {dataset_path}")
    
    # Check if environment is activated
    current_python = sys.executable
    is_venv_active = str(venv_python) in current_python
    
    print(f"\n🚀 ENVIRONMENT STATUS:")
    if is_venv_active:
        print(f"   ✅ Virtual environment is ACTIVE")
        print(f"   ✅ Current Python: {current_python}")
        
        # Check key packages
        try:
            import torch
            import transformers
            import accelerate
            import peft
            import bitsandbytes
            import pandas
            
            print(f"\n📦 INSTALLED PACKAGES:")
            print(f"   ✅ PyTorch: {torch.__version__}")
            print(f"   ✅ Transformers: {transformers.__version__}")
            print(f"   ✅ Accelerate: {accelerate.__version__}")
            print(f"   ✅ PEFT: {peft.__version__}")
            print(f"   ✅ BitsAndBytes: {bitsandbytes.__version__}")
            print(f"   ✅ Pandas: {pandas.__version__}")
            
            # GPU Check
            if torch.cuda.is_available():
                print(f"\n🔥 GPU STATUS:")
                print(f"   ✅ CUDA Available: {torch.cuda.is_available()}")
                print(f"   ✅ CUDA Version: {torch.version.cuda}")
                print(f"   ✅ GPU Count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"   ✅ GPU {i}: {gpu_name} ({memory_total:.1f}GB)")
            else:
                print(f"\n❌ GPU STATUS: CUDA not available")
                
        except ImportError as e:
            print(f"   ❌ Package import error: {e}")
            
    else:
        print(f"   ⚠️  Virtual environment is NOT active")
        print(f"   ℹ️  Run: source activate_env.sh")
    
    # Next steps
    print(f"\n🎯 NEXT STEPS:")
    if not is_venv_active:
        print(f"   1. Activate environment: source activate_env.sh")
    print(f"   2. Analyze dataset: python scripts/analyze_excel_dataset.py")
    print(f"   3. Start development: jupyter notebook")
    print(f"   4. Begin training: python scripts/train_phi3.py")
    
    print(f"\n🎉 ENVIRONMENT SETUP COMPLETE!")
    print(f"📚 Read README.md for detailed usage instructions")

if __name__ == "__main__":
    check_environment()