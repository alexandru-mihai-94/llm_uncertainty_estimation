# Quick Start Guide

Get started with Factoscope in 5 minutes!

## Prerequisites

- Python 3.8+
- 16GB RAM minimum
- Meta-Llama-3-8B model downloaded

## Installation (2 minutes)

```bash
# 1. Navigate to repository
cd /Users/alexandrumihai/Documents/llm_uncertainty_estimation/factoscope-uncertainty

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test installation
python tests/test_installation.py
```

## First Run (3 minutes)

### Option A: Use Pre-trained Model

If you have a trained model already:

```bash
python examples/example_single_question.py
```

### Option B: Train from Scratch

```bash
# Quick training (100 samples, 10 epochs)
python scripts/train_factoscope.py \
    --max_samples 100 \
    --epochs 10 \
    --device cpu
```

## Interactive Testing

```bash
python scripts/test_inference.py --interactive
```

Then enter questions:
```
Question: What is the capital of France?
Answer: Paris
Confidence: 87.2%
Prediction: CORRECT
```

## Next Steps

1. **Full Training**: See [USAGE.md](USAGE.md#training)
2. **Batch Analysis**: See [USAGE.md](USAGE.md#batch-analysis)
3. **Custom Datasets**: See [data/README.md](data/README.md)

## Common Commands

```bash
# Train on all datasets
python scripts/train_factoscope.py --epochs 30

# Test on MMLU
python scripts/test_external_datasets.py \
    --dataset ../datasets/mmlu_10k_answers.json --limit 100

# Generate visualizations
python scripts/batch_analysis.py \
    --dataset data/test_questions.json --plot_dir ./plots
```

## Troubleshooting

**Problem**: Import errors
```bash
# Solution: Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Problem**: Out of memory
```bash
# Solution: Reduce batch size
python scripts/train_factoscope.py --batch_size 2
```

**Problem**: Model not found
```bash
# Solution: Check path
ls models/Meta-Llama-3-8B/
# If empty, download model (see SETUP.md)
```

## Getting Help

- **Documentation**: README.md, SETUP.md, USAGE.md
- **Examples**: `examples/` directory
- **Issues**: GitHub Issues

## Summary

✅ Installation complete
✅ Model ready
✅ Examples running
✅ Ready to use!

For detailed usage, see [USAGE.md](USAGE.md).
