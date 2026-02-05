# Setup Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU also works)
- At least 8GB RAM (16GB+ recommended for larger models)

## Installation Steps

### 1. Clone or Download the Repository

```bash
git clone https://github.com/yourusername/llm-uncertainty-mwe.git
cd llm-uncertainty-mwe
```

Or download and extract the ZIP file.

### 2. Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (for model inference)
- Transformers (HuggingFace library)
- NumPy (numerical computing)
- Additional utilities (tqdm, matplotlib, scikit-learn)

### 4. Verify Installation

Run the test script to verify everything is working:

```bash
python3 test_installation.py
```

This will:
- Check all dependencies are installed
- Verify imports work correctly
- Test basic functionality (without loading a large model)

## Running Examples

### Single Question Example

```bash
python3 example_single.py
```

This will load a model and estimate uncertainty for one question.

**Note**: First run will download the model (~6GB for Llama 3.2 3B), which may take several minutes.

### Batch Processing Example

```bash
python3 example_batch.py
```

Processes multiple questions and shows aggregate statistics.

### Dataset Evaluation Example

```bash
python3 example_dataset.py
```

Shows how to evaluate predictions against ground truth.

## Memory Requirements

| Model | RAM Required | GPU VRAM |
|-------|-------------|----------|
| Llama 3.2 3B | 8GB+ | 6GB+ |
| Phi-3 Mini | 8GB+ | 6GB+ |
| Llama 3.1 8B | 16GB+ | 12GB+ |

If you encounter out-of-memory errors, try:
1. Using CPU instead: `device="cpu"` (slower but uses regular RAM)
2. Reducing `max_new_tokens` parameter
3. Using a smaller model

## Troubleshooting

### Import Errors

If you get import errors:

```bash
pip install --upgrade transformers torch numpy
```

### CUDA Errors

If you have GPU but get CUDA errors:

```bash
# Check PyTorch CUDA version
python3 -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
# See: https://pytorch.org/get-started/locally/
```

### Model Download Issues

If model download fails or is interrupted:

```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/

# Try downloading again
python3 example_single.py
```

### Out of Memory

If you run out of memory:

```python
# Use CPU mode
estimator = UncertaintyEstimator(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    device="cpu"  # Force CPU usage
)
```

Or reduce generation length:

```python
result = estimator.estimate(
    question="Your question?",
    max_new_tokens=20  # Reduced from default 50
)
```

## Using Different Models

The code supports various models. To change the model:

```python
from uncertainty_estimator import UncertaintyEstimator

# Use Phi-3 Mini instead
estimator = UncertaintyEstimator(
    model_name="microsoft/Phi-3-mini-4k-instruct"
)

# Or Gemma 2 2B
estimator = UncertaintyEstimator(
    model_name="google/gemma-2-2b-it"
)
```

Make sure you have accepted the model's license on HuggingFace and are logged in:

```bash
huggingface-cli login
```

## Configuration

### Adjusting Feature Weights

```python
estimator = UncertaintyEstimator(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    token_weight=0.6,        # 60% weight to token features
    activation_weight=0.4    # 40% weight to activation features
)
```

### Custom Layer Selection

```python
estimator = UncertaintyEstimator(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    target_layers=[5, 10, 15, 20, 25]  # Sample these specific layers
)
```

### Temperature Settings

```python
# Greedy decoding (deterministic)
result = estimator.estimate(question, temperature=0.0)

# Low temperature (mostly deterministic)
result = estimator.estimate(question, temperature=0.1)

# High temperature (more random)
result = estimator.estimate(question, temperature=1.0)
```

## Next Steps

1. **Run the examples** to get familiar with the API
2. **Try different models** to compare performance
3. **Evaluate on your own datasets** using `example_dataset.py` as a template
4. **Experiment with feature weights** to optimize for your use case

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review example scripts for usage patterns
3. Open an issue on GitHub with:
   - Python version (`python3 --version`)
   - Installed package versions (`pip freeze`)
   - Full error message
   - Minimal code to reproduce the issue

## Citation

If you use this code in your research, please cite:

```bibtex
@software{llm_uncertainty_mwe,
  author = {Mihai, Alexandru},
  title = {LLM Uncertainty Estimation - Minimal Working Example},
  year = {2025},
  institution = {OIST}
}
```
