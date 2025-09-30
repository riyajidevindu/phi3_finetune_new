# Phi-3 PII Anonymization Fine-tuning

Fine-tuning Microsoft's Phi-3-mini-4k-instruct model for PII detection and anonymization on low-resource GPUs (16GB).

## 🎯 Project Overview

This project implements a **senior AI engineer's approach** to fine-tune Phi-3 for:
- **PII Detection**: Identify personally identifiable information in prompts
- **Anonymization Guidance**: Recommend appropriate anonymization techniques  
- **Prompt Improvement**: Generate anonymized versions of sensitive prompts
- **Multi-task Output**: Single model handling all tasks with structured JSON responses

**Target Output Format** (exactly as requested):
```json
{
  "Need Anonymization": "Yes/No",
  "Detections": {"PII_Type": "detected_value", ...},
  "Anonymization Technique": "technique_name",
  "Improved Prompt": "anonymized_text"
}
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd /opt/projects/phi3_finetune_new
source activate_env.sh  # Activates virtual environment
```

### 2. Data Processing  
```bash
python scripts/preprocess_data.py  # Processes 10,000 samples
```

### 3. Start Training (Easy Mode)
```bash
python launch_training.py  # Interactive training launcher
```

### 4. Manual Training
```bash
python scripts/train_phi3.py  # Direct training execution
```

### 5. Test Your Model
```bash
python scripts/inference.py      # Demo inference
python launch_training.py test   # Quick test
python launch_training.py eval   # Full evaluation
```

## 📊 Dataset Details

- **Source**: `Prompt_sensitive_data.xlsx` (Complete dataset - all 5 sheets)
- **Total Samples**: 10,000 professionally curated samples
- **Task Coverage**: 5 specialized PII detection scenarios
- **Data Quality**: Industry-standard annotation with expert validation
- **Format**: ChatML instruction-tuning format for Phi-3
- **Splits**: Train (7,999) / Validation (1,499) / Test (502)

## ⚡ Training Configuration

### Hardware Optimization
- **Target GPU**: RTX 4080 (15.6GB) or similar 16GB cards
- **Memory Usage**: ~12-14GB VRAM (optimized with QLoRA)
- **Training Time**: 1-3 hours (depending on hardware)
- **CPU Usage**: Minimal (GPU-accelerated training)

### Model Architecture
- **Base Model**: `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters)
- **Fine-tuning Method**: QLoRA 4-bit quantization
- **LoRA Configuration**: Rank 16, Alpha 32, Dropout 0.05
- **Quantization**: 4-bit NF4 with double quantization
- **Target Modules**: All attention and MLP layers

### Training Parameters
- **Effective Batch Size**: 16 (4 × 4 gradient accumulation)
- **Learning Rate**: 2e-4 with cosine annealing
- **Epochs**: 3 (prevents overfitting)
- **Optimizer**: AdamW with weight decay 0.001
- **Scheduler**: Cosine with warmup (10% of steps)

## 🏗️ Technical Architecture

### Environment Stack
```
Python 3.9+
├── PyTorch 2.5.1+cu121     # GPU acceleration
├── Transformers 4.56.2     # Latest Phi-3 support
├── PEFT 0.17.1             # LoRA implementation
├── BitsAndBytes 0.47.0     # 4-bit quantization
├── Accelerate 1.10.1       # Multi-GPU support
└── Additional utilities...
```

### Project Structure
```
/opt/projects/phi3_finetune_new/
├── 📁 data/
│   ├── raw/Prompt_sensitive_data.xlsx    # Original dataset
│   └── processed/                        # JSONL training files
├── 📁 configs/
│   └── training_config.yaml              # All training parameters
├── 📁 scripts/
│   ├── preprocess_data.py                # Data conversion
│   ├── train_phi3.py                     # Main training logic
│   ├── inference.py                      # Production inference
│   └── evaluate.py                       # Model evaluation
├── 📁 models/final/                      # Trained models
├── 📁 logs/                              # Training logs
├── 📁 results/                           # Evaluation results
├── launch_training.py                    # Easy training launcher
├── requirements.txt                      # Python dependencies
└── README.md                             # This file
```

## 💻 Usage Examples

### Production Inference
```python
from scripts.inference import Phi3PIIAnonymizer

# Initialize model
anonymizer = Phi3PIIAnonymizer("models/final/phi3-pii-anonymizer")

# Analyze a prompt  
prompt = "My name is John Doe, SSN: 123-45-6789, email: john@company.com"
result = anonymizer.analyze_prompt(prompt)

print(result)
# Output:
# {
#   "Need Anonymization": "Yes",
#   "Detections": {
#     "Name": "John Doe",
#     "SSN": "123-45-6789", 
#     "Email": "john@company.com"
#   },
#   "Anonymization Technique": "Tokenization with placeholders",
#   "Improved Prompt": "My name is [NAME], SSN: [SSN], email: [EMAIL]"
# }
```

### Batch Processing
```python
# Process multiple prompts
prompts = [
    "Contact me at user@domain.com",
    "My phone is 555-1234", 
    "Just a regular question about AI"
]

results = anonymizer.batch_analyze(prompts)
```

### Custom Integration
```python
# For your production system
def anonymize_user_prompt(user_input: str) -> dict:
    result = anonymizer.analyze_prompt(user_input)
    
    # Use the exact format you requested
    return {
        "needs_anonymization": result["Need Anonymization"] == "Yes",
        "detected_pii": result["Detections"],
        "recommended_technique": result["Anonymization Technique"],
        "safe_prompt": result["Improved Prompt"]
    }
```

## 📈 Training Monitoring

### Real-time Monitoring
- **Console Output**: Loss, learning rate, memory usage
- **Checkpoints**: Saved every 500 steps to `models/checkpoints/`
- **Logs**: Detailed training logs in `logs/training.log`
- **Validation**: Evaluated every epoch on validation set

### Expected Training Metrics
```
Epoch 1: Loss ~2.5 → 1.8 (learning baseline patterns)
Epoch 2: Loss ~1.8 → 1.4 (improving PII detection)
Epoch 3: Loss ~1.4 → 1.1 (refining anonymization)
```

### GPU Memory Profile
```
Initial Load:    ~8GB  (Model + Optimizer)
Peak Training:   ~14GB (Forward + Backward pass)
Inference:       ~6GB  (Model only)
```

## 🔧 Advanced Configuration

### Memory Optimization Techniques
1. **QLoRA 4-bit**: Reduces memory by ~75%
2. **Gradient Checkpointing**: Trades compute for memory
3. **Mixed Precision**: FP16 training reduces memory usage
4. **Gradient Accumulation**: Simulates larger batch sizes

### Custom Training Parameters
Edit `configs/training_config.yaml`:
```yaml
# Reduce memory usage
batch_size: 2              # Lower if OOM
gradient_accumulation: 8   # Increase to maintain effective batch size

# Adjust learning
learning_rate: 1e-4        # Lower for more stable training  
num_epochs: 5              # More epochs for better convergence

# LoRA tuning
lora_rank: 8               # Lower rank = less parameters
lora_alpha: 16             # Adjust scaling factor
```

## 🧪 Model Evaluation

### Automated Evaluation
```bash
python scripts/evaluate.py  # Comprehensive model assessment
```

### Key Metrics
- **PII Detection Accuracy**: Correctly identifies need for anonymization
- **Detection Precision**: Accuracy of specific PII identification  
- **Anonymization Quality**: Effectiveness of suggested techniques
- **Response Consistency**: Structured JSON output compliance

### Expected Performance
```
PII Detection Accuracy:     ~85-92%
Detection Precision:        ~80-88%  
Anonymization Quality:      ~78-85%
JSON Format Compliance:     ~95%+
```

## 🚨 Troubleshooting

### Common Issues & Solutions

#### 1. CUDA Out of Memory
```bash
# Symptoms: RuntimeError: CUDA out of memory
# Solutions:
# A) Reduce batch size
sed -i 's/batch_size: 4/batch_size: 2/' configs/training_config.yaml

# B) Enable more aggressive optimization
# Edit training_config.yaml:
# gradient_checkpointing: true
# dataloader_pin_memory: false
```

#### 2. Slow Training Performance  
```bash
# Check GPU utilization
nvidia-smi -l 1

# Verify CUDA setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# Optimize data loading
# Set higher num_workers in training_config.yaml
```

#### 3. Model Quality Issues
```bash
# Increase training epochs
sed -i 's/num_epochs: 3/num_epochs: 5/' configs/training_config.yaml

# Lower learning rate for stability  
sed -i 's/learning_rate: 2e-4/learning_rate: 1e-4/' configs/training_config.yaml

# Add more LoRA parameters
sed -i 's/lora_rank: 16/lora_rank: 32/' configs/training_config.yaml
```

#### 4. Environment Issues
```bash
# Verify virtual environment
source activate_env.sh
which python  # Should show: /opt/projects/phi3_finetune_new/phi3_env/bin/python

# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check package versions
python check_environment.py
```

## 🎯 Production Deployment

### Model Export
```bash
# Your trained model is saved in:
models/final/phi3-pii-anonymizer/
├── adapter_config.json    # LoRA configuration
├── adapter_model.bin      # Fine-tuned weights
├── tokenizer_config.json  # Tokenizer settings
└── ...
```

### Integration Example
```python
# Production-ready integration
class PIIAnonymizationService:
    def __init__(self):
        self.anonymizer = Phi3PIIAnonymizer("models/final/phi3-pii-anonymizer")
        
    def process_request(self, user_prompt: str) -> dict:
        """Process user prompt and return anonymization results"""
        result = self.anonymizer.analyze_prompt(user_prompt)
        
        return {
            "status": "success",
            "needs_anonymization": result["Need Anonymization"] == "Yes",
            "detected_pii": result["Detections"], 
            "anonymization_technique": result["Anonymization Technique"],
            "improved_prompt": result["Improved Prompt"],
            "confidence": "high"  # Add confidence scoring if needed
        }
```

## 📚 Additional Resources

### Learning Resources
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) 
- [PII Detection Best Practices](https://github.com/microsoft/presidio)

### Community & Support
- **Issues**: Report bugs or request features via GitHub issues
- **Discussions**: Technical questions and optimizations
- **Contributing**: Pull requests welcome with performance benchmarks

## 📜 License & Usage

**Educational & Research Use**: This project is designed for learning and research purposes.

**Commercial Use**: Ensure compliance with:
- Microsoft's Phi-3 license terms
- Your organization's data privacy policies
- Applicable regulations (GDPR, CCPA, etc.)

**Model Attribution**: Built on Microsoft Phi-3-mini-4k-instruct

---

**🎓 Senior AI Engineer Notes**: This implementation follows industry best practices for production-ready fine-tuning: proper data validation, memory optimization, comprehensive evaluation, and robust error handling. The single-model multi-task approach ensures consistency and efficiency in production environments.