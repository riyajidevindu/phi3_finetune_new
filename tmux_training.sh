#!/bin/bash
# Tmux Session Manager for Phi-3 Training
# Persistent training sessions that survive disconnections

set -e

# Configuration
PROJECT_DIR="/opt/projects/phi3_finetune_new"
SESSION_NAME="phi3-training"
WINDOW_NAME="training"
ENV_FILE=".env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check tmux
    if ! command -v tmux &> /dev/null; then
        print_error "tmux is not installed"
        echo "Install with: sudo apt-get install tmux"
        exit 1
    fi
    
    # Check project directory
    if [ ! -d "$PROJECT_DIR" ]; then
        print_error "Project directory not found: $PROJECT_DIR"
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f "$PROJECT_DIR/$ENV_FILE" ]; then
        print_warning ".env file not found. Creating from .env.example..."
        if [ -f "$PROJECT_DIR/.env.example" ]; then
            cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
            print_warning "Please edit $PROJECT_DIR/.env with your WANDB_API_KEY"
            print_warning "Run: nano $PROJECT_DIR/.env"
        else
            print_error ".env.example file not found"
            exit 1
        fi
    fi
    
    print_success "Dependencies check passed"
}

create_training_session() {
    print_status "Creating tmux training session..."
    
    # Kill existing session if it exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        print_warning "Existing session found. Killing it..."
        tmux kill-session -t "$SESSION_NAME"
    fi
    
    # Create new session
    tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR"
    
    # Rename window
    tmux rename-window -t "$SESSION_NAME:0" "$WINDOW_NAME"
    
    # Setup environment in the session
    tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" "source activate_env.sh" C-m
    
    # Wait a moment for environment activation
    sleep 2
    
    print_success "Training session created: $SESSION_NAME"
}

start_training() {
    print_status "Starting training in tmux session..."
    
    # Check if processed data exists
    tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" "if [ ! -d 'data/processed' ] || [ -z \"\$(ls -A data/processed)\" ]; then echo 'Processing data first...'; python scripts/preprocess_data.py; fi" C-m
    
    # Wait for potential data processing
    sleep 3
    
    # Start training
    tmux send-keys -t "$SESSION_NAME:$WINDOW_NAME" "python scripts/train_phi3.py" C-m
    
    print_success "Training started in session: $SESSION_NAME"
    print_status "Use 'tmux attach -t $SESSION_NAME' to view progress"
}

monitor_session() {
    print_status "Monitoring options:"
    echo "1. Attach to session: tmux attach -t $SESSION_NAME"
    echo "2. View session list: tmux list-sessions"
    echo "3. Kill session: tmux kill-session -t $SESSION_NAME"
    echo "4. Detach from session: Ctrl+B, then D"
    echo ""
    
    # Check if session is running
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        print_success "Training session is active"
        
        # Ask if user wants to attach
        read -p "Do you want to attach to the training session now? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Attaching to session. Use Ctrl+B, then D to detach."
            sleep 2
            tmux attach -t "$SESSION_NAME"
        fi
    else
        print_error "Training session is not running"
    fi
}

show_session_status() {
    print_status "Session Status:"
    
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        print_success "Session '$SESSION_NAME' is ACTIVE"
        
        # Get session info
        echo ""
        echo "Session Details:"
        tmux list-sessions -F "#{session_name}: #{session_windows} windows, created #{session_created_string}" | grep "$SESSION_NAME"
        
        echo ""
        echo "Windows in session:"
        tmux list-windows -t "$SESSION_NAME" -F "#{window_index}: #{window_name} (#{window_panes} panes)"
        
        echo ""
        echo "Recent activity (last 10 lines):"
        tmux capture-pane -t "$SESSION_NAME:$WINDOW_NAME" -p | tail -10
        
    else
        print_warning "Session '$SESSION_NAME' is NOT running"
    fi
}

setup_wandb_key() {
    print_status "Setting up WANDB API Key..."
    
    if [ -f "$PROJECT_DIR/.env" ]; then
        # Check if WANDB_API_KEY is already set
        if grep -q "^WANDB_API_KEY=your_wandb_api_key_here" "$PROJECT_DIR/.env"; then
            print_warning "WANDB_API_KEY is not configured in .env file"
            echo ""
            echo "Please get your WANDB API key from: https://wandb.ai/authorize"
            echo ""
            read -p "Enter your WANDB API key: " wandb_key
            
            if [ ! -z "$wandb_key" ]; then
                # Replace the placeholder with actual key
                sed -i "s/^WANDB_API_KEY=your_wandb_api_key_here/WANDB_API_KEY=$wandb_key/" "$PROJECT_DIR/.env"
                print_success "WANDB API key updated in .env file"
            else
                print_warning "No API key provided. Training will run in offline mode."
            fi
        else
            print_success "WANDB_API_KEY is already configured"
        fi
    else
        print_error ".env file not found"
    fi
}

show_help() {
    echo "Phi-3 Training Tmux Manager"
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start      - Create session and start training"
    echo "  create     - Create tmux session only (no training)"
    echo "  attach     - Attach to existing session"
    echo "  status     - Show session status"
    echo "  stop       - Stop training session"
    echo "  restart    - Restart training session"
    echo "  setup-key  - Setup WANDB API key"
    echo "  logs       - View recent training logs"
    echo "  help       - Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 start           # Start complete training session"
    echo "  $0 setup-key       # Configure WANDB API key first"
    echo "  $0 attach          # Attach to running session"
    echo "  $0 status          # Check if training is running"
}

# Main script logic
case "${1:-help}" in
    "start")
        check_dependencies
        setup_wandb_key
        create_training_session
        start_training
        monitor_session
        ;;
    "create")
        check_dependencies
        create_training_session
        echo "Session created. Use '$0 start' to begin training."
        ;;
    "attach")
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            print_status "Attaching to session: $SESSION_NAME"
            tmux attach -t "$SESSION_NAME"
        else
            print_error "No active session found. Use '$0 start' to create one."
        fi
        ;;
    "status")
        show_session_status
        ;;
    "stop")
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            print_status "Stopping training session..."
            tmux kill-session -t "$SESSION_NAME"
            print_success "Session stopped"
        else
            print_warning "No active session to stop"
        fi
        ;;
    "restart")
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            print_status "Restarting training session..."
            tmux kill-session -t "$SESSION_NAME"
        fi
        sleep 2
        $0 start
        ;;
    "setup-key")
        setup_wandb_key
        ;;
    "logs")
        if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            print_status "Recent training logs:"
            tmux capture-pane -t "$SESSION_NAME:$WINDOW_NAME" -p
        else
            print_warning "No active session. Checking log files..."
            if [ -d "$PROJECT_DIR/logs" ]; then
                find "$PROJECT_DIR/logs" -name "*.log" -exec tail -20 {} \;
            else
                print_error "No logs found"
            fi
        fi
        ;;
    "help"|*)
        show_help
        ;;
esac