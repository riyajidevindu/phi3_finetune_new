# WANDB and Tmux Setup Guide

## 🔑 Setting up Weights & Biases (WANDB)

### 1. Get Your API Key
1. Go to https://wandb.ai/authorize
2. Sign in or create an account
3. Copy your API key

### 2. Configure Environment Variables
```bash
# Edit your .env file
nano .env

# Replace this line:
WANDB_API_KEY=your_wandb_api_key_here

# With your actual key:
WANDB_API_KEY=your_actual_api_key_here

# Optional: Set your username/team
WANDB_ENTITY=your_username_or_team_name
```

### 3. Test WANDB Setup
```bash
# Activate environment
source activate_env.sh

# Test wandb
python -c "import wandb; print('WANDB ready!')"
```

## 🖥️  Tmux Session Management

### Why Use Tmux for Training?
- **Persistent Sessions**: Training continues even if you disconnect
- **Remote Training**: Perfect for SSH connections
- **Session Recovery**: Reconnect to ongoing training anytime
- **Background Execution**: Training runs independently

### Basic Tmux Commands

#### Start Training Session
```bash
# Start complete training in tmux
./tmux_training.sh start

# Just create session (no training yet)
./tmux_training.sh create
```

#### Monitor Training
```bash
# Check if training is running
./tmux_training.sh status

# Attach to running session
./tmux_training.sh attach

# View recent logs
./tmux_training.sh logs
```

#### Session Control
```bash
# Stop training session
./tmux_training.sh stop

# Restart training
./tmux_training.sh restart
```

#### Inside Tmux Session
- **Detach**: `Ctrl+B`, then `D` (training continues)
- **Scroll**: `Ctrl+B`, then `[`, use arrow keys, `q` to quit
- **Kill Session**: `Ctrl+B`, then `:kill-session`

## 🚀 Complete Training Workflow

### Method 1: Quick Start (Interactive)
```bash
cd /opt/projects/phi3_finetune_new
source activate_env.sh
python launch_training.py
```

### Method 2: Production Setup (Tmux - Recommended)
```bash
cd /opt/projects/phi3_finetune_new

# Setup WANDB key (one time only)
./tmux_training.sh setup-key

# Start persistent training
./tmux_training.sh start
```

### Method 3: Manual Tmux Setup
```bash
# Create tmux session
tmux new-session -s phi3-training -c /opt/projects/phi3_finetune_new

# In tmux session:
source activate_env.sh
python scripts/train_phi3.py

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -s phi3-training
```

## 📊 Monitoring Training Progress

### Real-time Monitoring
```bash
# Watch training in real-time
./tmux_training.sh attach

# Check status without attaching
./tmux_training.sh status

# View recent output
./tmux_training.sh logs
```

### WANDB Dashboard
1. Go to https://wandb.ai
2. Navigate to your project: `phi3-pii-anonymization`
3. Monitor loss curves, GPU usage, and metrics

### Local Logs
```bash
# View training logs
tail -f logs/training.log

# Check GPU usage
nvidia-smi -l 1
```

## 🔧 Troubleshooting

### WANDB Issues
```bash
# Test WANDB connection
python -c "import wandb; wandb.login()"

# Run in offline mode
export WANDB_MODE=offline
python scripts/train_phi3.py
```

### Tmux Issues
```bash
# List all sessions
tmux list-sessions

# Kill stuck session
tmux kill-session -t phi3-training

# Restart tmux service
sudo systemctl restart tmux
```

### Environment Issues
```bash
# Verify environment
source activate_env.sh
which python
python check_environment.py

# Reinstall dependencies
pip install -r requirements.txt
```

## 💡 Pro Tips

### For Remote Training
```bash
# SSH with tmux forwarding
ssh -t user@server "cd /opt/projects/phi3_finetune_new && tmux attach -t phi3-training"

# Or start remote training
ssh user@server "./opt/projects/phi3_finetune_new/tmux_training.sh start"
```

### For Long Training Sessions
- Always use tmux for training > 1 hour
- Set up WANDB for remote monitoring
- Check disk space before starting
- Monitor GPU temperature during training

### Memory Optimization
```bash
# If you get OOM errors, reduce batch size in configs/training_config.yaml:
batch_size: 2  # Instead of 4
gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

## 📋 Pre-Training Checklist

- [ ] WANDB API key configured in `.env`
- [ ] Virtual environment activated
- [ ] Data processed (`data/processed/` exists)
- [ ] GPU available (`nvidia-smi` works)
- [ ] Tmux installed (`tmux -V`)
- [ ] Enough disk space (>10GB free)
- [ ] Internet connection for model download

## 🎯 Training Timeline

### Expected Duration (RTX 4080)
- **Data Loading**: 2-3 minutes
- **Model Download**: 5-10 minutes (first time)
- **Training**: 1-3 hours (3 epochs)
- **Total**: ~1.5-3.5 hours

### Checkpoints
- Saved every 500 steps to `models/checkpoints/`
- Final model saved to `models/final/phi3-pii-anonymizer`
- WANDB logs all metrics automatically

Ready to start training! 🚀