#!/bin/bash
# Setup script for Phi-3 Fine-tuning Environment
# Optimized for 16GB GPU with memory constraints

set -e  # Exit on any error

echo "🚀 Setting up Phi-3 Fine-tuning Environment"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "phi3_env" ]; then
    echo "📁 Creating virtual environment..."
    python3 -m venv phi3_env
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source phi3_env/bin/activate

# Upgrade pip first
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch first (with CUDA support for GPU)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other core ML libraries
echo "🧠 Installing core ML libraries..."
pip install transformers[torch] accelerate peft bitsandbytes

# Install data processing libraries
echo "📊 Installing data processing libraries..."
pip install pandas numpy openpyxl xlrd scikit-learn datasets

# Install training utilities
echo "🏋️  Installing training utilities..."
pip install trl evaluate wandb tensorboard

# Install development tools
echo "🛠️  Installing development tools..."
pip install jupyter notebook matplotlib seaborn tqdm click python-dotenv pyyaml

# Install monitoring tools
echo "📊 Installing monitoring tools..."
pip install psutil py3nvml gpustat

# Install text processing
echo "📝 Installing text processing tools..."
pip install nltk spacy regex jsonlines

# Install testing and quality tools
echo "🧪 Installing testing and quality tools..."
pip install pytest black flake8 mypy

# Verify critical installations
echo "✅ Verifying installations..."

python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

python -c "
try:
    import transformers
    print(f'Transformers version: {transformers.__version__}')
    
    import accelerate
    print(f'Accelerate version: {accelerate.__version__}')
    
    import peft
    print(f'PEFT version: {peft.__version__}')
    
    import bitsandbytes
    print(f'BitsAndBytes version: {bitsandbytes.__version__}')
    
    import pandas
    print(f'Pandas version: {pandas.__version__}')
    
    print('✅ All core libraries installed successfully!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1
"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/processed
mkdir -p models/checkpoints
mkdir -p models/final
mkdir -p logs
mkdir -p results
mkdir -p notebooks
mkdir -p tests
mkdir -p src/phi3_finetune
mkdir -p configs

echo "🎉 Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source phi3_env/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
echo "📊 GPU Memory Check:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available - make sure NVIDIA drivers are installed"
echo ""
echo "🚀 Ready to start fine-tuning Phi-3!"