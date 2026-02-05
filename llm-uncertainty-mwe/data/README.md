# Dataset Directory

This directory contains example datasets for testing and evaluating the uncertainty estimator.

## Files

### mmlu_10k_answers.json (3.3 MB)
- **Source**: MMLU (Massive Multitask Language Understanding)
- **Size**: 10,000 question-answer pairs
- **Format**: JSON array of objects
- **Use case**: Full evaluation, parameter tuning, model comparison

### mmlu_sample_100.json (33 KB)
- **Source**: First 100 questions from MMLU
- **Size**: 100 question-answer pairs
- **Format**: JSON array of objects
- **Use case**: Quick testing, debugging, demonstrations

## Format

Each entry contains:
```json
{
  "id": 0,
  "question": "Where is the Louvre museum?",
  "answer": "Paris"
}
```

## Dataset Statistics

### Full Dataset (10,000 questions)
- Diverse subjects: STEM, humanities, social sciences
- Difficulty: Varies from basic facts to complex reasoning
- Answer types: Short phrases, names, technical terms

### Usage Examples

#### Load Full Dataset
```python
import json

with open('data/mmlu_10k_answers.json', 'r') as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} questions")
```

#### Load Sample Dataset
```python
import json

with open('data/mmlu_sample_100.json', 'r') as f:
    dataset = json.load(f)

# Quick testing with smaller dataset
for item in dataset[:5]:
    print(f"Q: {item['question']}")
    print(f"A: {item['answer']}\n")
```

## Use Cases

### 1. Quick Testing
Use `mmlu_sample_100.json` for rapid iteration:
- Test new features
- Debug code changes
- Verify installation
- Quick parameter experiments

### 2. Full Evaluation
Use `mmlu_10k_answers.json` for comprehensive analysis:
- Model comparison
- Parameter optimization
- Calibration studies
- Statistical analysis

### 3. Training/Calibration Split

```python
import json
import random

with open('data/mmlu_10k_answers.json', 'r') as f:
    data = json.load(f)

# Shuffle
random.seed(42)
random.shuffle(data)

# Split: 70% train, 30% test
split_idx = int(0.7 * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"Training: {len(train_data)} questions")
print(f"Testing: {len(test_data)} questions")
```

### 4. Stratified Sampling

```python
import json

with open('data/mmlu_10k_answers.json', 'r') as f:
    data = json.load(f)

# Sample every Nth question for balanced representation
sample_rate = 10
stratified_sample = data[::sample_rate]

print(f"Sampled {len(stratified_sample)} questions")
```

## Important Notes

### Answer Matching

The dataset contains ground truth answers, but matching LLM outputs to these answers requires careful handling:

**Challenges:**
- LLMs may phrase answers differently (e.g., "Paris" vs "Paris, France")
- Extra context in responses (e.g., "The answer is Paris" vs "Paris")
- Spelling variations or abbreviations

**Recommended Approaches:**
1. **Substring matching**: Check if answer is contained in response
2. **Fuzzy matching**: Use string similarity (e.g., Levenshtein distance)
3. **Semantic matching**: Use embeddings to compare meaning
4. **Manual verification**: For small datasets, verify a sample manually

**Example:**
```python
def simple_match(llm_answer, true_answer):
    """Simple substring matching (case-insensitive)."""
    return true_answer.lower() in llm_answer.lower()

def fuzzy_match(llm_answer, true_answer, threshold=0.8):
    """Fuzzy string matching."""
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None,
                                 llm_answer.lower(),
                                 true_answer.lower()).ratio()
    return similarity >= threshold
```

### Small Model Performance

Based on empirical testing:
- **Llama 3.2 3B**: ~15-20% accuracy on this dataset
- **Phi-3 Mini**: ~12-18% accuracy on this dataset
- **Expected**: 25% random chance (if these were multiple choice)

Small models (<4B parameters) struggle with MMLU's knowledge demands, resulting in:
- High error rates
- Class imbalance (~85% incorrect)
- Predictor bias toward "incorrect" predictions

### Best Practices

1. **Start with sample**: Use `mmlu_sample_100.json` first
2. **Verify matching**: Check answer matching logic on a few examples
3. **Consider difficulty**: Some questions are intentionally hard
4. **Balance classes**: Consider oversampling correct answers if training
5. **Cross-validation**: Use multiple train/test splits for robust evaluation

## Dataset Source

This dataset is derived from MMLU (Massive Multitask Language Understanding):
- Original paper: [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
- Converted from multiple-choice to question-answer format
- See `convert_json.py` in parent directory for conversion script

## License

Please refer to the original MMLU dataset license and cite appropriately if using in research.
