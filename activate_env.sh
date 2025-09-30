#!/bin/bash

# Activate Virtual Environment Script
# Use: source activate_env.sh

PROJECT_ROOT="/opt/projects/phi3_finetune_new"

if [ ! -d "$PROJECT_ROOT/phi3_env" ]; then
    echo "❌ Virtual environment not found. Please run setup_environment.sh first."
    return 1 2>/dev/null || exit 1
fi

# Activate the virtual environment
source "$PROJECT_ROOT/phi3_env/bin/activate"

# Set environment variables
export PROJECT_ROOT="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/models/cache"
export HF_HOME="$PROJECT_ROOT/models/cache"
export TOKENIZERS_PARALLELISM=false

# GPU Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_USE_CUDA_DSA=1

echo "🚀 Phi-3 Fine-tuning Environment Activated!"
echo "📁 Project Root: $PROJECT_ROOT"
echo "🐍 Python: $(which python)"
echo "💾 GPU Memory: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) MB free"
echo ""
echo "Available commands:"
echo "  • python scripts/analyze_excel_dataset.py  - Analyze dataset"
echo "  • python scripts/preprocess_data.py       - Preprocess training data"
echo "  • python scripts/train_phi3.py            - Start fine-tuning"
echo "  • jupyter notebook                         - Start Jupyter"
echo "  • deactivate                               - Deactivate environment"