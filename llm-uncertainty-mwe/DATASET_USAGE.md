# Using the MMLU Dataset

This guide shows how to use the included MMLU dataset for various tasks.

## Quick Start

### Load and Inspect

```python
import json

# Load sample dataset (100 questions)
with open('data/mmlu_sample_100.json', 'r') as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} questions")

# Show first question
print(dataset[0])
# Output: {"id": 0, "question": "Where is the Louvre museum?", "answer": "Paris"}
```

### Run Evaluation

```bash
# Quick test with sample dataset
python3 example_mmlu.py

# This will:
# - Load the sample dataset (100 questions)
# - Process first 20 questions
# - Compare LLM answers with ground truth
# - Display comprehensive metrics
```

## Use Cases

### 1. Quick Testing

Use the sample dataset (100 questions) for rapid development:

```python
from uncertainty_estimator import UncertaintyEstimator
import json

# Load sample
with open('data/mmlu_sample_100.json', 'r') as f:
    dataset = json.load(f)

# Initialize
estimator = UncertaintyEstimator(
    model_name="meta-llama/Llama-3.2-3B-Instruct"
)

# Test on first 5 questions
questions = [item['question'] for item in dataset[:5]]
results = estimator.estimate_batch(questions)

# Quick check
for i, (result, item) in enumerate(zip(results, dataset[:5]), 1):
    print(f"{i}. {item['question'][:50]}...")
    print(f"   Expected: {item['answer']}")
    print(f"   Got: {result['answer']}")
    print(f"   Confidence: {result['confidence_score']:.1%}\n")
```

### 2. Parameter Tuning

Test different configurations to find optimal settings:

```bash
python3 example_parameter_tuning.py
```

This script:
- Tests different feature weight combinations
- Compares temperature settings
- Identifies best performing configuration
- Provides tuning recommendations

### 3. Full Dataset Evaluation

For comprehensive analysis, use the full 10K dataset:

```python
import json
from uncertainty_estimator import UncertaintyEstimator

# Load full dataset
with open('data/mmlu_10k_answers.json', 'r') as f:
    full_dataset = json.load(f)

print(f"Loaded {len(full_dataset)} questions")

# Process in batches to save memory
batch_size = 100
estimator = UncertaintyEstimator(
    model_name="meta-llama/Llama-3.2-3B-Instruct"
)

all_results = []
for i in range(0, len(full_dataset), batch_size):
    batch = full_dataset[i:i+batch_size]
    questions = [item['question'] for item in batch]
    
    results = estimator.estimate_batch(questions)
    all_results.extend(results)
    
    print(f"Processed {len(all_results)}/{len(full_dataset)}")

# Analyze results...
```

**Note**: Processing 10K questions takes several hours on CPU, ~30-60 minutes on GPU.

### 4. Train/Test Split

Create calibration and evaluation sets:

```python
import json
import random

# Load dataset
with open('data/mmlu_10k_answers.json', 'r') as f:
    data = json.load(f)

# Shuffle with fixed seed for reproducibility
random.seed(42)
random.shuffle(data)

# Split: 70% train, 30% test
split_idx = int(0.7 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"Training: {len(train_data)} questions")
print(f"Testing: {len(test_data)} questions")

# Save splits
with open('data/mmlu_train.json', 'w') as f:
    json.dump(train_data, f, indent=2)

with open('data/mmlu_test.json', 'w') as f:
    json.dump(test_data, f, indent=2)
```

### 5. Difficulty-Based Sampling

Sample questions based on certain criteria:

```python
import json

with open('data/mmlu_10k_answers.json', 'r') as f:
    data = json.load(f)

# Filter by answer length (proxy for complexity)
simple_questions = [d for d in data if len(d['answer'].split()) <= 2]
complex_questions = [d for d in data if len(d['answer'].split()) > 5]

print(f"Simple questions: {len(simple_questions)}")
print(f"Complex questions: {len(complex_questions)}")

# Or filter by question length
short_questions = [d for d in data if len(d['question']) < 100]
long_questions = [d for d in data if len(d['question']) > 200]

print(f"Short questions: {len(short_questions)}")
print(f"Long questions: {len(long_questions)}")
```

## Answer Matching Strategies

The dataset provides ground truth answers, but matching LLM outputs requires careful handling.

### 1. Simple Substring Matching

```python
def simple_match(llm_answer, true_answer):
    """Check if true answer is contained in LLM answer."""
    return true_answer.lower() in llm_answer.lower()

# Example
llm_ans = "The capital of France is Paris."
true_ans = "Paris"
print(simple_match(llm_ans, true_ans))  # True
```

### 2. Fuzzy Matching

```python
from difflib import SequenceMatcher

def fuzzy_match(llm_answer, true_answer, threshold=0.7):
    """Use string similarity for matching."""
    similarity = SequenceMatcher(
        None, 
        llm_answer.lower(), 
        true_answer.lower()
    ).ratio()
    return similarity >= threshold

# Example
llm_ans = "Pari"  # Typo or truncated
true_ans = "Paris"
print(fuzzy_match(llm_ans, true_ans, threshold=0.7))  # True
```

### 3. Token-Based Matching

```python
def token_match(llm_answer, true_answer):
    """Match if any token overlaps."""
    llm_tokens = set(llm_answer.lower().split())
    true_tokens = set(true_answer.lower().split())
    return len(llm_tokens & true_tokens) > 0

# Example
llm_ans = "The answer is Paris, France"
true_ans = "Paris"
print(token_match(llm_ans, true_ans))  # True
```

### 4. Multiple Strategy Combination

```python
def robust_match(llm_answer, true_answer):
    """Try multiple matching strategies."""
    # Strategy 1: Exact substring
    if simple_match(llm_answer, true_answer):
        return True
    
    # Strategy 2: Fuzzy matching
    if fuzzy_match(llm_answer, true_answer, threshold=0.8):
        return True
    
    # Strategy 3: Token overlap
    if token_match(llm_answer, true_answer):
        return True
    
    return False
```

See `example_mmlu.py` for complete implementation.

## Performance Expectations

Based on empirical testing with small models:

| Model | Accuracy on MMLU | Notes |
|-------|------------------|-------|
| Llama 3.2 3B | 15-20% | Below random (25%) |
| Phi-3 Mini | 12-18% | Struggles with knowledge |
| Random Guess | 25% | 4-choice baseline |

**Key Takeaways:**
1. Small models (<4B params) have insufficient capacity for MMLU
2. This creates ~85% incorrect vs ~15% correct imbalance
3. Predictors trained on this are biased toward "incorrect"
4. Consider larger models (7B+) for better base accuracy

## Advanced Usage

### Cross-Validation

```python
import json
import numpy as np

with open('data/mmlu_10k_answers.json', 'r') as f:
    data = json.load(f)

# 5-fold cross-validation
n_folds = 5
fold_size = len(data) // n_folds

for fold in range(n_folds):
    test_start = fold * fold_size
    test_end = (fold + 1) * fold_size
    
    test_set = data[test_start:test_end]
    train_set = data[:test_start] + data[test_end:]
    
    print(f"Fold {fold+1}: Train={len(train_set)}, Test={len(test_set)}")
    
    # Evaluate on this fold...
```

### Stratified Sampling

```python
import json
import random

with open('data/mmlu_10k_answers.json', 'r') as f:
    data = json.load(f)

# Sample every Nth question for balanced representation
n = 10  # Sample rate
stratified = data[::n]

print(f"Sampled {len(stratified)} questions (every {n}th)")
```

### Confidence-Based Analysis

```python
# Group results by confidence levels
high_conf = [r for r in results if r['confidence_score'] >= 0.65]
med_conf = [r for r in results if 0.40 <= r['confidence_score'] < 0.65]
low_conf = [r for r in results if r['confidence_score'] < 0.40]

print(f"High confidence: {len(high_conf)} predictions")
print(f"Medium confidence: {len(med_conf)} predictions")
print(f"Low confidence: {len(low_conf)} predictions")

# Check accuracy within each group
# (requires ground truth matching)
```

## Tips for Best Results

1. **Start Small**: Use `mmlu_sample_100.json` first
2. **Verify Matching**: Check a few examples manually to ensure answer matching works
3. **Balance Classes**: Consider oversampling correct answers if training classifiers
4. **Multiple Metrics**: Track accuracy, F1, calibration (ECE), not just one
5. **Cross-Validate**: Use multiple train/test splits for robust evaluation
6. **Consider Model Size**: Small models (<4B) will struggle; consider 7B+ for better results

## Getting Help

For issues or questions:
1. Check `data/README.md` for dataset details
2. See `example_mmlu.py` for complete evaluation example
3. Review `example_parameter_tuning.py` for optimization strategies
4. Open an issue on GitHub with your specific question
