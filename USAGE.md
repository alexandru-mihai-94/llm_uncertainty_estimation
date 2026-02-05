# Usage Guide

Complete guide to using the Factoscope uncertainty estimation framework.

## About This Implementation

This is an **extended and refactored version** of the original Factoscope:

- **Original**: https://github.com/JenniferHo97/llm_factoscope (Ho, He, Lyu, Ye et al., 2024)
- **Core methodology**: Extracted from original implementation
- **New features**: Modern LLM support, extended datasets, batch analysis, modular architecture

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training](#training)
3. [Inference](#inference)
4. [Batch Analysis](#batch-analysis)
5. [External Testing](#external-testing)
6. [Library API](#library-api)
7. [Advanced Usage](#advanced-usage)

## Quick Start

### Single Question

```python
from factoscope import FactoscopeInference

engine = FactoscopeInference(
    model_path='./models/Meta-Llama-3-8B',
    factoscope_model_path='./factoscope_output/best_factoscope_model.pt',
    processed_data_path='./factoscope_output/processed_data.h5'
)

result = engine.predict_confidence("What is the capital of France?")
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Run Examples

```bash
# Single question
python examples/example_single_question.py

# Batch processing
python examples/example_batch.py
```

## Training

### Full Training Pipeline

```bash
python scripts/train_factoscope.py \
    --model_path ./models/Meta-Llama-3-8B \
    --dataset_dir ./llm_factoscope-main/dataset_train \
    --output_dir ./factoscope_output \
    --max_samples 500 \
    --epochs 30 \
    --device cpu
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | `./models/Meta-Llama-3-8B` | Path to LLM |
| `--dataset_dir` | `./llm_factoscope-main/dataset_train` | Training datasets |
| `--output_dir` | `./factoscope_output` | Output directory |
| `--max_samples` | `500` | Samples per dataset |
| `--epochs` | `30` | Training epochs |
| `--batch_size` | `8` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--support_size` | `100` | Support set size |
| `--device` | `cpu` | Device (cpu/cuda) |

### Training Steps

1. **Data Collection** (Step 1)
   - Loads LLM
   - Processes each dataset
   - Collects hidden states, probabilities, ranks
   - Saves to HDF5 files

2. **Preprocessing** (Step 2)
   - Normalizes hidden states
   - Transforms ranks
   - Balances correct/false samples
   - Creates single H5 file

3. **Training** (Step 3)
   - Creates triplet dataset
   - Trains with triplet loss
   - Evaluates on test set
   - Saves best model

### Skip Steps

```bash
# Skip data collection (use existing features)
python scripts/train_factoscope.py --skip_collection

# Skip training (only collect data)
python scripts/train_factoscope.py --skip_training
```

### Output Files

```
factoscope_output/
├── features/                      # Per-dataset features
│   ├── athlete_country/
│   │   ├── correct_data.h5
│   │   ├── false_data.h5
│   │   ├── unrelative_data.h5
│   │   └── metadata.json
│   └── ...
├── processed_data.h5              # Combined preprocessed data
└── best_factoscope_model.pt       # Trained model
```

## Inference

### Interactive Mode

```bash
python scripts/test_inference.py --interactive
```

Example session:
```
Question: What is the capital of France?

Answer: Paris
First Token: Paris
Confidence: 0.87 (weighted)
Simple Confidence: 0.80
Prediction: CORRECT
Nearest Correct Distance: 0.3421
Nearest False Distance: 0.7856
Distance Ratio: 0.4354 (favors correct)
Top-10 Neighbors: 8 correct, 2 false
Token Rank: 0, Top Probability: 0.9823
```

### Batch Mode

```bash
python scripts/test_inference.py \
    --questions "What is 2+2?" "Who wrote Hamlet?" \
    --output results.json
```

### From JSON File

```bash
python scripts/test_inference.py \
    --questions_file test_questions.json \
    --output results.json
```

### Result Format

```json
{
  "prompt": "What is the capital of France?",
  "answer": "Paris",
  "first_token": "Paris",
  "confidence": 0.872,
  "simple_confidence": 0.800,
  "prediction": "correct",
  "nearest_correct_distance": 0.3421,
  "nearest_false_distance": 0.7856,
  "distance_ratio": 0.4354,
  "top_k_neighbors": {
    "correct": 8,
    "false": 2
  },
  "rank": 0,
  "top_prob": 0.9823
}
```

## Batch Analysis

### Run Analysis

```bash
python scripts/batch_analysis.py \
    --dataset ./llm_factoscope-main/dataset_train/athlete_country_dataset.json \
    --output batch_results.json \
    --plot_dir ./plots \
    --limit 100
```

### Generated Plots

1. **confidence_distribution.png** - Histogram and box plot of confidence scores
2. **prediction_accuracy.png** - Bar chart and pie chart of predictions
3. **confidence_vs_prediction.png** - Violin plots by prediction type
4. **top_prob_vs_confidence.png** - Scatter plot of token prob vs confidence
5. **distance_heatmap.png** - Nearest neighbor distances
6. **rank_distribution.png** - Token rank distribution
7. **neighbor_breakdown.png** - k-NN composition per question

### From Existing Results

```bash
python scripts/batch_analysis.py \
    --results_file batch_results.json \
    --plot_dir ./plots
```

## External Testing

### Test on MMLU

```bash
python scripts/test_external_datasets.py \
    --dataset ./datasets/mmlu_10k_answers.json \
    --limit 100 \
    --output_dir ./external_test_results
```

### Test on Multiple Datasets

```bash
python scripts/test_external_datasets.py \
    --test_all \
    --dataset_dir ./datasets \
    --limit 50
```

### Interactive Selection

```bash
python scripts/test_external_datasets.py
```

Will prompt:
```
Available datasets:

  1. mmlu_10k_answers.json (3.3 MB)
  2. hellaswag_dataset.json (2.1 MB)
  3. cosmosqa_dataset.json (1.8 MB)

Select dataset number (or 'all' to test all):
```

### Output

```
external_test_results/run_TIMESTAMP/
├── mmlu_10k_answers_results.json
├── mmlu_10k_answers_plots/
│   └── [7 plot PNG files]
├── hellaswag_results.json
├── hellaswag_plots/
└── test_summary.json
```

## Library API

### Data Collection

```python
from factoscope import FactDataCollector

collector = FactDataCollector(
    model_path='./models/Meta-Llama-3-8B',
    device='cuda'
)

# Single question
result = collector.generate_and_collect(
    prompt="What is the capital of France?",
    max_new_tokens=10
)

# Process dataset
collector.process_dataset(
    dataset_path='./data/questions.json',
    output_dir='./features',
    max_samples=100
)
```

### Preprocessing

```python
from factoscope import FactDataPreprocessor

preprocessor = FactDataPreprocessor(feature_dir='./factoscope_output')

stats = preprocessor.prepare_training_data(
    dataset_paths=['./features/dataset1', './features/dataset2'],
    output_file='./processed_data.h5'
)
```

### Training

```python
from factoscope import FactoscopeModel, FactoscopeTrainer, FactoscopeDataset
import torch
from torch.utils.data import DataLoader

# Create model
model = FactoscopeModel(
    num_layers=33,
    hidden_dim=4096,
    emb_dim=32,
    final_dim=64
)

# Create trainer
trainer = FactoscopeTrainer(model, device='cuda')

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(30):
    loss = trainer.train_epoch(train_loader, optimizer)
    metrics = trainer.evaluate(test_loader, support_loader)
```

### Inference

```python
from factoscope import FactoscopeInference

engine = FactoscopeInference(
    model_path='./models/Meta-Llama-3-8B',
    factoscope_model_path='./best_factoscope_model.pt',
    processed_data_path='./processed_data.h5',
    device='cuda'
)

# Single question
result = engine.predict_confidence("Question here")

# Batch
results = engine.batch_predict(["Q1", "Q2", "Q3"])
```

## Advanced Usage

### Custom Neuron Selection

Modify data collection to use different neuron selection strategies:

```python
collector = FactDataCollector(model_path, device='cuda')

# Modify unrelative tokens
collector.unrelative_tokens.add('custom_word')
```

### Custom Preprocessing

```python
preprocessor = FactDataPreprocessor(feature_dir)

# Custom normalization
data, mean, std = preprocessor.normalize_data(
    data,
    mean=custom_mean,
    std=custom_std
)

# Custom rank transformation
ranks_transformed = preprocessor.transform_ranks(ranks)
```

### Model Architecture Changes

```python
from factoscope.model import FactoscopeModel

# Larger embedding dimensions
model = FactoscopeModel(
    num_layers=33,
    hidden_dim=4096,
    emb_dim=64,  # Increased from 32
    final_dim=128  # Increased from 64
)
```

### Custom k-NN Parameters

```python
# In inference.py, modify predict_confidence():
k = 20  # Use 20 nearest neighbors instead of 10
topk_distances, topk_indices = torch.topk(distances, k, largest=False)
```

### Different LLMs

The framework works with any LLaMA-like model. Adjust paths:

```python
engine = FactoscopeInference(
    model_path='./models/Llama-2-7B',  # Different model
    factoscope_model_path='./factoscope_llama2.pt',
    processed_data_path='./processed_llama2.h5'
)
```

## Tips & Best Practices

### Training

1. **Start small**: Use `--max_samples 100` for testing
2. **Monitor overfitting**: Check if train loss << test loss
3. **Adjust learning rate**: Try 0.0001 or 0.01 if not converging
4. **Use GPU**: Training is ~5x faster on GPU

### Inference

1. **Warm up**: First prediction is slow (model loading)
2. **Batch when possible**: More efficient than one-by-one
3. **Cache support set**: Precompute embeddings for speed
4. **Monitor distances**: Very high distances indicate out-of-distribution

### Data

1. **Balance datasets**: Equal correct/false samples
2. **Diverse questions**: Include multiple domains
3. **Quality over quantity**: 500 good samples > 5000 noisy ones

## Troubleshooting

### Low Accuracy

- Increase training epochs
- Use more training data
- Check data balance (correct vs false)
- Verify model loaded correctly

### High Confidence Errors

- Increase support set size
- Use more k-NN neighbors
- Retrain on more diverse data

### Slow Inference

- Use GPU (`--device cuda`)
- Reduce support set size
- Cache model in memory

### Memory Errors

- Reduce batch size
- Use CPU for inference
- Process datasets one at a time
- Reduce `max_samples`

## See Also

- [README.md](README.md) - Project overview
- [SETUP.md](SETUP.md) - Installation guide
- [data/README.md](data/README.md) - Dataset information
