# LLM Uncertainty Estimation - Minimal Working Example

A minimal implementation for estimating answer correctness in small language models using token-level statistics and layer activation patterns.

## Overview

This repository demonstrates how to use internal model signals (token probabilities and hidden layer activations) to predict whether an LLM's answer is likely correct or incorrect.

**Key Finding**: When tested on small models (3-4B parameters) with MMLU dataset, base accuracy was <20%. The predictors trained on this data are biased toward predicting "incorrect" due to extreme class imbalance (~85% incorrect examples).

## Features

- **Token-level statistics**: Confidence, margins, ranking patterns
- **Activation-based features**: Layer norms, cross-layer consistency, trajectory analysis
- **Model support**: Llama, Phi-3, Gemma, and other transformer models
- **Simple API**: Easy to integrate into existing pipelines

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-uncertainty-mwe.git
cd llm-uncertainty-mwe

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 test_installation.py
```

## Included Dataset

The repository includes the **MMLU (Massive Multitask Language Understanding)** dataset:
- **Full dataset**: `data/mmlu_10k_answers.json` (10,000 questions)
- **Sample dataset**: `data/mmlu_sample_100.json` (100 questions for quick testing)

See `data/README.md` for detailed dataset documentation.

## Quick Start

```python
from uncertainty_estimator import UncertaintyEstimator

# Initialize estimator
estimator = UncertaintyEstimator(
    model_name="meta-llama/Llama-3.2-3B-Instruct"
)

# Estimate uncertainty for a question
result = estimator.estimate(
    question="What is the capital of France?"
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence_score']:.2%}")
print(f"Prediction: {result['predicted_correct']}")
```

## Usage Examples

### Single Question

```python
from uncertainty_estimator import UncertaintyEstimator

estimator = UncertaintyEstimator(model_name="meta-llama/Llama-3.2-3B-Instruct")

result = estimator.estimate(
    question="Who wrote Romeo and Juliet?",
    temperature=0.1
)

print(result)
```

### Batch Processing

```python
from uncertainty_estimator import UncertaintyEstimator

estimator = UncertaintyEstimator(model_name="meta-llama/Llama-3.2-3B-Instruct")

questions = [
    "What is 2+2?",
    "Who was the first US president?",
    "What is the speed of light?"
]

results = estimator.estimate_batch(questions)

for r in results:
    print(f"Q: {r['question']}")
    print(f"A: {r['answer']}")
    print(f"Confidence: {r['confidence_score']:.2%}\n")
```

### Using the MMLU Dataset

```python
import json
from uncertainty_estimator import UncertaintyEstimator

# Load dataset
with open('data/mmlu_sample_100.json', 'r') as f:
    dataset = json.load(f)

# Process questions
estimator = UncertaintyEstimator(model_name="meta-llama/Llama-3.2-3B-Instruct")
questions = [item['question'] for item in dataset[:10]]
results = estimator.estimate_batch(questions)

# Compare with ground truth
for result, item in zip(results, dataset[:10]):
    print(f"Expected: {item['answer']}")
    print(f"Got: {result['answer']}")
    print(f"Confidence: {result['confidence_score']:.2%}\n")
```

### Extract Features Only

```python
from feature_extractor import FeatureExtractor

extractor = FeatureExtractor(model_name="meta-llama/Llama-3.2-3B-Instruct")

features = extractor.extract_features(
    question="What is the capital of Japan?",
    answer_ids=generated_ids,
    hidden_states=hidden_states,
    logits=logits
)

print("Token features:", features['token'])
print("Activation features:", features['activation'])
```

## How It Works

### 1. Token-Level Features

Extracted from the probability distribution during generation:

- **Confidence**: Probability assigned to selected tokens
- **Margin**: Difference between top-1 and top-2 token probabilities
- **Ranking**: Position of selected token in probability distribution
- **Temporal patterns**: How confidence changes across the sequence

### 2. Activation Features

Extracted from transformer hidden states:

- **Activation norms**: L2 norm of activation vectors
- **Cross-layer consistency**: Similarity between different layers
- **Trajectory consistency**: Smoothness of token-to-token transitions
- **Sparsity**: Concentration of activations

### 3. Combination

Features are combined using empirically-derived weights to produce a confidence score between 0 and 1.

## Configuration

### Model-Specific Layer Sampling

Different models use different layer configurations:

```python
estimator = UncertaintyEstimator(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    target_layers=[1, 5, 9, 13, 17, 21, 25]  # Custom layers
)
```

### Feature Weights

Adjust the relative importance of token vs activation features:

```python
estimator = UncertaintyEstimator(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    token_weight=0.4,      # 40% token features
    activation_weight=0.6  # 60% activation features
)
```

## Limitations

1. **Small model accuracy**: 3-4B parameter models achieve <20% accuracy on MMLU (below random)
2. **Class imbalance**: Predictors biased toward "incorrect" due to ~85% incorrect training examples
3. **Computational overhead**: Extracting hidden states ~2x inference time
4. **Dataset specific**: Results based on MMLU; may not generalize to other domains

## Project Structure

```
llm-uncertainty-mwe/
├── README.md                      # This file
├── SETUP.md                       # Detailed setup guide
├── STRUCTURE.md                   # Repository organization
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
│
├── Core Library:
├── uncertainty_estimator.py       # Main estimator class
├── feature_extractor.py           # Feature extraction utilities
├── utils.py                       # Helper functions
│
├── Examples:
├── test_installation.py           # Installation verification
├── example_single.py              # Single question example
├── example_batch.py               # Batch processing example
├── example_dataset.py             # Dataset evaluation example
├── example_mmlu.py                # MMLU dataset evaluation
├── example_parameter_tuning.py    # Parameter optimization
│
└── data/                          # Included datasets
    ├── README.md                  # Dataset documentation
    ├── mmlu_10k_answers.json      # Full MMLU dataset (10K questions)
    └── mmlu_sample_100.json       # Sample dataset (100 questions)
```

## Performance

Tested on MMLU dataset:

- **Base accuracy**: 15-20% (Llama 3.2 3B, Phi-3 Mini)
- **ROC AUC**: ~0.85 (but contextualized by extreme class imbalance)
- **Temperature independence**: <3% variation across temperature 0.1-1.0 (artifact of floor performance)

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

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Contact

Alexandru Mihai - OIST

## Acknowledgments

Based on research exploring token-level and activation-based predictors for LLM answer correctness.
