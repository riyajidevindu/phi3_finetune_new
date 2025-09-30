#!/usr/bin/env python3
"""
Training Launcher for Phi-3 PII Anonymization
Simplified training execution with monitoring
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu():
    """Check GPU availability and memory"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("🎮 GPU Status:")
            print(result.stdout)
            return True
        else:
            print("❌ No GPU detected")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - GPU may not be available")
        return False

def check_environment():
    """Check if we're in the right environment"""
    current_dir = Path.cwd()
    expected_dir = Path("/opt/projects/phi3_finetune_new")
    
    if current_dir != expected_dir:
        print(f"❌ Please run from {expected_dir}")
        print(f"   Current directory: {current_dir}")
        return False
    
    # Check if virtual environment is activated
    if not os.environ.get('VIRTUAL_ENV'):
        print("❌ Virtual environment not activated")
        print("   Run: source activate_env.sh")
        return False
    
    # Check if processed data exists
    data_dir = Path("data/processed")
    if not data_dir.exists() or not list(data_dir.glob("*.jsonl")):
        print("❌ Processed data not found")
        print("   Run: python scripts/preprocess_data.py")
        return False
    
    print("✅ Environment checks passed")
    return True

def start_training():
    """Start the training process"""
    
    print("🚀 STARTING PHI-3 PII ANONYMIZATION TRAINING")
    print("=" * 50)
    
    # Environment checks
    if not check_environment():
        return False
    
    # GPU check
    check_gpu()
    
    print("\n📋 Training Configuration:")
    print("- Model: microsoft/Phi-3-mini-4k-instruct")
    print("- Method: QLoRA 4-bit fine-tuning")
    print("- Task: Multi-task PII detection & anonymization")
    print("- GPU Memory Optimization: Enabled")
    print("- Expected Duration: 1-3 hours")
    
    # Confirm start
    response = input("\n🤔 Start training? (y/n): ").lower().strip()
    if response != 'y':
        print("Training cancelled")
        return False
    
    print("\n🏃‍♂️ Starting training...")
    print("💡 You can monitor progress in the terminal output")
    print("📊 Training logs will be saved to logs/")
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Start training
    try:
        cmd = [sys.executable, "scripts/train_phi3.py"]
        
        # Run with output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in process.stdout:
            print(line, end='')
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code == 0:
            print("\n🎉 Training completed successfully!")
            print("📁 Model saved to: models/final/phi3-pii-anonymizer")
            print("🧪 Test the model: python scripts/inference.py")
            return True
        else:
            print(f"\n❌ Training failed with return code: {return_code}")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        return False

def quick_test():
    """Quick test of the trained model"""
    model_path = Path("models/final/phi3-pii-anonymizer")
    
    if not model_path.exists():
        print("❌ Trained model not found - train first")
        return
    
    print("\n🧪 Running quick inference test...")
    
    try:
        subprocess.run([sys.executable, "scripts/inference.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Inference test failed")
    except KeyboardInterrupt:
        print("⏹️ Test interrupted")

def setup_wandb_key():
    """Setup WANDB API key if not configured"""
    env_file = Path(".env")
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
        
        if "WANDB_API_KEY=your_wandb_api_key_here" in content:
            print("\n🔑 WANDB API Key Setup")
            print("=" * 30)
            print("Get your API key from: https://wandb.ai/authorize")
            
            api_key = input("Enter your WANDB API key (or press Enter to skip): ").strip()
            
            if api_key:
                # Replace placeholder with actual key
                new_content = content.replace(
                    "WANDB_API_KEY=your_wandb_api_key_here",
                    f"WANDB_API_KEY={api_key}"
                )
                
                with open(env_file, 'w') as f:
                    f.write(new_content)
                
                print("✅ WANDB API key configured")
                return True
            else:
                print("⚠️  Skipping WANDB setup - training will run in offline mode")
                return False
    return True

def main():
    """Main launcher"""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            quick_test()
            return
        elif sys.argv[1] == "analyze":
            subprocess.run([sys.executable, "scripts/analyze_training.py"])
            return
        elif sys.argv[1] == "eval":
            subprocess.run([sys.executable, "scripts/evaluate.py"])
            return
        elif sys.argv[1] == "tmux":
            print("�️  For tmux training, use: ./tmux_training.sh start")
            return
    
    print("�🔧 PHI-3 PII ANONYMIZATION TRAINER")
    print("=" * 40)
    print("Training Options:")
    print("  python launch_training.py        - Interactive training")
    print("  ./tmux_training.sh start         - Persistent tmux session")
    print()
    print("Other Commands:")
    print("  python launch_training.py test     - Test trained model")
    print("  python launch_training.py eval     - Evaluate model")
    print("  python launch_training.py analyze  - Analyze training results")
    print("  ./tmux_training.sh status          - Check tmux session")
    print()
    
    # Setup WANDB key if needed
    setup_wandb_key()
    
    # Ask user preference
    print("🤔 Choose training method:")
    print("1. Interactive training (current terminal)")
    print("2. Persistent tmux session (recommended for long training)")
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "2":
        print("\n🖥️  Starting tmux session...")
        try:
            subprocess.run(["./tmux_training.sh", "start"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to start tmux session")
            print("   Falling back to interactive training...")
            start_training()
    else:
        success = start_training()
        
        if success:
            # Offer to test
            response = input("\n🧪 Run inference test? (y/n): ").lower().strip()
            if response == 'y':
                quick_test()

if __name__ == "__main__":
    main()