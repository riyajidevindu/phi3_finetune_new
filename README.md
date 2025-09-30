# Phi-3 Fine-tuning Project

A comprehensive fine-tuning pipeline for Microsoft Phi-3-mini-4k-instruct optimized for PII detection and anonymization tasks on 16GB GPU systems.

## 🎯 Project Overview

This project fine-tunes the Phi-3-mini-4k-instruct model on a multi-task dataset containing 10,000 samples across 5 specialized domains:

- **Medical & Demographic PII** (2,000 samples)
- **Location & Contact Info** (2,000 samples)  
- **Financial & Identification Info** (2,000 samples)
- **Employment Education & Social** (2,000 samples)
- **Internal Document Prompts** (2,000 samples)

## 🔧 Key Features

- **Memory Optimized**: QLoRA 4-bit quantization for 16GB GPU
- **Multi-task Learning**: PII detection, extraction, and anonymization
- **Production Ready**: Complete MLOps pipeline with monitoring
- **Space Efficient**: Virtual environment in project directory

## 📊 Model Tasks

1. **PII Detection**: Binary classification (Yes/No)
2. **PII Extraction**: Structured JSON output with PII categories
3. **Prompt Anonymization**: Generate privacy-safe prompts

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Activate the environment
source activate_env.sh

# Or manually activate
source phi3_env/bin/activate
```

### 2. Data Analysis
```bash
python scripts/analyze_excel_dataset.py
```

### 3. Data Preprocessing
```bash
python scripts/preprocess_data.py
```

### 4. Start Training
```bash
python scripts/train_phi3.py
```

### 5. Jupyter Development
```bash
jupyter notebook
```

## 📁 Project Structure

```
phi3_finetune_new/
├── 📊 dataset/                 # Raw Excel dataset (10,000 samples)
├── 📈 data/processed/          # Preprocessed training data
├── 🤖 models/
│   ├── cache/                  # Model cache directory
│   ├── checkpoints/           # Training checkpoints
│   └── final/                 # Final trained models
├── 📝 scripts/                # Training and analysis scripts
├── 📔 notebooks/              # Jupyter notebooks
├── ⚙️ configs/                # Configuration files
├── 📊 results/                # Training results and metrics
├── 🧪 tests/                  # Unit tests
├── 🐍 phi3_env/               # Virtual environment
└── 📋 logs/                   # Training logs
```

## 💾 Hardware Requirements

- **GPU**: 16GB VRAM (RTX 4080/4090, V100, A100)
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ free space
- **CUDA**: 12.1+ compatible

## 🔧 Configuration

### Training Parameters
- **Model**: microsoft/Phi-3-mini-4k-instruct (3.8B params)
- **Quantization**: QLoRA 4-bit (NF4)
- **Batch Size**: 1-2 per GPU
- **Gradient Accumulation**: 8-16 steps
- **Learning Rate**: 2e-4
- **Max Sequence Length**: 2048 tokens
- **LoRA Rank**: 16

### Memory Optimization
- **Estimated VRAM Usage**: 12-14GB with QLoRA
- **Peak Memory**: ~15GB during training
- **Gradient Checkpointing**: Enabled
- **DataLoader Workers**: 4

## 📊 Dataset Statistics

- **Total Samples**: 10,000
- **PII Samples**: 5,024 (50.2%)
- **Non-PII Samples**: 4,976 (49.8%)
- **Average Sequence Length**: ~150 tokens
- **Max Sequence Length**: 2,048 tokens

## 🏋️ Training Pipeline

1. **Data Loading**: Multi-sheet Excel processing
2. **Preprocessing**: Tokenization and formatting
3. **Model Loading**: Phi-3 with QLoRA adapters
4. **Training**: Multi-task supervised fine-tuning
5. **Evaluation**: Validation on held-out set
6. **Monitoring**: Weights & Biases integration

## 📈 Monitoring & Logging

- **Weights & Biases**: Experiment tracking
- **TensorBoard**: Local monitoring
- **GPU Monitoring**: Real-time VRAM usage
- **Checkpoint Saving**: Every 100 steps

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_data_processing.py
```

## 📝 Usage Examples

### Quick Inference
```python
from src.phi3_finetune.inference import Phi3PIIDetector

detector = Phi3PIIDetector("models/final/phi3-pii-detector")
result = detector.detect_pii("My name is John Doe and my SSN is 123-45-6789")
print(result)
```

### Batch Processing
```python
from src.phi3_finetune.batch_processor import BatchProcessor

processor = BatchProcessor("models/final/phi3-pii-detector")
results = processor.process_file("input_data.csv")
```

## 🔒 Privacy & Security

- **No Data Retention**: Model trained locally
- **Secure Processing**: PII detection without storage
- **Anonymization**: Safe prompt generation
- **Compliance Ready**: GDPR/CCPA compatible

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**Out of Memory Error**
```bash
# Reduce batch size in configs/training_config.yaml
batch_size: 1
gradient_accumulation_steps: 16
```

**CUDA Out of Memory**
```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

**Environment Issues**
```bash
# Rebuild environment
rm -rf phi3_env
./setup_environment.sh
```

## 📚 References

- [Microsoft Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)

## 🏆 Acknowledgments

- Microsoft for the Phi-3 model family
- Hugging Face for the transformers library
- QLoRA authors for efficient fine-tuning techniques

---

**Happy Fine-tuning! 🚀**