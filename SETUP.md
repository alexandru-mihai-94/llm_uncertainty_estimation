# Setup Guide

This guide walks you through setting up the Factoscope uncertainty estimation framework.

## About This Implementation

This is an extended and refactored version of the original Factoscope implementation:

- **Original Repository**: https://github.com/JenniferHo97/llm_factoscope
- **Extended Version**: Updated for modern LLMs (LLaMA-3), extended datasets, modular architecture
- **Compatibility**: Works with Transformers 4.30+, PyTorch 2.0+

## Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.7+ (for GPU training, optional for CPU inference)
- **Disk Space**: ~30GB (for model and data)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: 16GB+ VRAM for training (T4, V100, A100)

## Installation Steps

### 1. Clone Repository

```bash
git clone git@github.com:alexandru-mihai-94/llm_uncertainty_estimation.git
cd llm_uncertainty_estimation/factoscope-uncertainty
```

### 2. Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n factoscope python=3.10
conda activate factoscope
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download LLM Model

This framework requires Meta-Llama-3-8B (or compatible models).

**Option A: Hugging Face Hub (Recommended)**

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login (requires access token)
huggingface-cli login

# Download model
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
           AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B', cache_dir='./models'); \
           AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B', cache_dir='./models')"
```

**Option B: Manual Download**

1. Request access: https://huggingface.co/meta-llama/Meta-Llama-3-8B
2. Download model files
3. Place in `./models/Meta-Llama-3-8B/`

### 5. Download Training Datasets

**Option A: Clone Original Factoscope Datasets**

```bash
# Clone the original Factoscope repository for training datasets
git clone https://github.com/JenniferHo97/llm_factoscope.git

# The datasets are located in:
# llm_factoscope/dataset_train/
```

**Option B: Download from Paper Supplementary Materials**

Visit the Factoscope paper and download the supplementary datasets.

**Option C: Use Custom Datasets**

See `data/README.md` for information on creating custom datasets and using external sources (MMLU, HellaSwag, etc.).

### 6. Verify Installation

```bash
python tests/test_installation.py
```

Expected output:
```
✓ PyTorch installed
✓ Transformers installed
✓ All dependencies available
✓ CUDA available (if GPU present)
✓ Installation successful!
```

## Directory Structure After Setup

```
factoscope-uncertainty/
├── models/
│   └── Meta-Llama-3-8B/          # Downloaded LLM
│       ├── config.json
│       ├── tokenizer.json
│       └── pytorch_model.bin
│
├── llm_factoscope-main/          # Training datasets
│   └── dataset_train/
│       ├── athlete_country_dataset.json
│       ├── book_author_dataset.json
│       └── ...
│
├── factoscope_output/            # Generated during training
│   ├── features/                 # Collected internal states
│   ├── processed_data.h5         # Preprocessed training data
│   └── best_factoscope_model.pt  # Trained model
│
└── [other project files]
```

## Configuration

### Environment Variables

```bash
# Optional: Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Optional: Set cache directory
export TRANSFORMERS_CACHE=./models
```

### Memory Optimization

For machines with limited RAM:

**1. Use CPU inference**
```bash
python scripts/test_inference.py --device cpu
```

**2. Reduce batch size**
```python
# In training scripts
--batch_size 4  # instead of 8 or 16
```

**3. Reduce max_samples**
```bash
python scripts/train_factoscope.py --max_samples 100
```

## Common Issues

### Issue: CUDA Out of Memory

**Solution**:
```bash
# Use CPU
python scripts/train_factoscope.py --device cpu

# Or reduce batch size
python scripts/train_factoscope.py --batch_size 2
```

### Issue: Model Download Fails

**Solution**:
1. Ensure Hugging Face access token is valid
2. Check internet connection
3. Try manual download option

### Issue: Dataset Not Found

**Solution**:
```bash
# Verify dataset paths
ls llm_factoscope-main/dataset_train/

# If missing, download from:
# [Add actual dataset source URL]
```

### Issue: Import Errors

**Solution**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or check Python version
python --version  # Should be 3.8+
```

## Troubleshooting

### Check GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
```

### Test Model Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('./models/Meta-Llama-3-8B')
tokenizer = AutoTokenizer.from_pretrained('./models/Meta-Llama-3-8B')
print("✓ Model loaded successfully")
```

### Verify H5 File Access

```python
import h5py
with h5py.File('./factoscope_output/processed_data.h5', 'r') as f:
    print(f"Keys: {list(f.keys())}")
    print(f"Shape: {f['hidden_states'].shape}")
```

## Performance Tuning

### Training Speed

- **GPU**: ~1-2 hours for 500 samples/dataset, 30 epochs
- **CPU**: ~6-10 hours for same configuration

### Memory Usage

- **Training**: 16GB RAM + 12GB VRAM (GPU)
- **Inference**: 8GB RAM + 8GB VRAM (GPU) or 16GB RAM (CPU)

## Next Steps

After successful installation:

1. **Train a Model**: See [USAGE.md](USAGE.md#training)
2. **Run Examples**: `python examples/example_single_question.py`
3. **Test Inference**: `python scripts/test_inference.py --interactive`

## Getting Help

- **Documentation**: See USAGE.md for detailed usage
- **Issues**: https://github.com/alexandru-mihai-94/llm_uncertainty_estimation/issues
- **Examples**: Check `examples/` directory
