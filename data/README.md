# Dataset Information

This directory contains information about datasets used for training and testing Factoscope.

## Important: Dataset Attribution

### Original Factoscope Datasets

The core training datasets are from the **original Factoscope repository**:

- **Source**: https://github.com/JenniferHo97/llm_factoscope
- **Authors**: Jennifer Ho, Jinwen He, and contributors
- **Paper**: "Factoscope: Uncovering LLMs' Factual Discernment through Inner States"

These datasets contain factual questions across multiple domains and were carefully curated for the original research.

### Location

Original Factoscope training datasets should be placed in:
```
llm_factoscope-main/dataset_train/
```

Clone the original repository to obtain them:
```bash
git clone https://github.com/JenniferHo97/llm_factoscope.git
```

### Available Datasets

1. **athlete_country_dataset.json** - Athlete nationality questions
2. **book_author_dataset.json** - Book authorship questions
3. **city_country_dataset.json** - City location questions
4. **movie_name_director_dataset.json** - Movie director questions
5. **pantheon_country_dataset.json** - Historical figure countries
6. **pantheon_occupation_dataset.json** - Historical figure occupations
7. **final_nobel_category_dataset.json** - Nobel prize categories
8. **year_olympicscity_dataset.json** - Olympic host cities
9. **movie_name_year_dataset.json** - Movie release years
10. **final_song_artist_dataset.json** - Song artist questions

### Dataset Format

Each dataset file contains JSON array of question objects:

```json
[
  {
    "index": 0,
    "prompt": "Question: Inger Anne Frøysedal is a citizen of which country?\nAnswer:",
    "answer": "Norway"
  },
  {
    "index": 1,
    "prompt": "Question: Michael Winter is a citizen of which country?\nAnswer:",
    "answer": "Canada"
  }
]
```

### Download

**Option 1: Clone from original repository (Recommended)**
```bash
git clone https://github.com/JenniferHo97/llm_factoscope.git
```

**Option 2: From paper supplementary materials**

Visit the Factoscope paper and download the supplementary datasets.

**Note**: All original training datasets are credited to the original Factoscope authors. This repository extends support to additional dataset formats but does not claim creation of the core training data.

## Test Datasets

### External Datasets (Newly Added)

**This extended implementation adds support for multiple external dataset formats:**

These datasets are **NOT part of the original Factoscope repository** and represent new functionality:

1. **MMLU** (Massive Multitask Language Understanding)
   - 10,000+ multiple-choice questions
   - Location: `../datasets/mmlu_10k_answers.json`
   - Format: `{"id": 0, "question": "...", "answer": "..."}`
   - Source: Public MMLU dataset, converted to Q&A format

2. **HellaSwag** - Common sense reasoning
   - Newly added format support

3. **CosmosQA** - Reading comprehension
   - Newly added format support

4. **TruthfulQA** - Factual accuracy
   - Newly added format support

**Key Addition**: The extended implementation includes automatic dataset format conversion to support diverse question-answer formats beyond the original Factoscope datasets.

### Creating Custom Datasets

Your dataset should follow this format:

```json
[
  {
    "id": 0,
    "prompt": "Question: What is the capital of France?\nAnswer:",
    "answer": "Paris"
  },
  {
    "id": 1,
    "prompt": "Question: Who wrote Hamlet?\nAnswer:",
    "answer": "William Shakespeare"
  }
]
```

Or simplified:

```json
[
  {
    "question": "What is the capital of France?",
    "answer": "Paris"
  }
]
```

The framework will automatically convert to prompt format.

### Example Creation Script

```python
import json

questions = [
    {"question": "What is 2+2?", "answer": "4"},
    {"question": "What color is the sky?", "answer": "blue"},
]

# Convert to prompt format
dataset = []
for i, q in enumerate(questions):
    dataset.append({
        "id": i,
        "prompt": f"Question: {q['question']}\nAnswer:",
        "answer": q['answer']
    })

# Save
with open('custom_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)
```

## Dataset Statistics

### Training Data (after collection)

Typical distribution after processing 500 samples per dataset:

| Dataset | Correct | False | Unrelative |
|---------|---------|-------|------------|
| athlete_country | 120 | 340 | 40 |
| book_author | 95 | 380 | 25 |
| city_country | 150 | 310 | 40 |
| movie_director | 110 | 360 | 30 |
| pantheon_country | 130 | 350 | 20 |

**Note**: "Unrelative" means the first token was an article, pronoun, or filler word.

### Balancing

During preprocessing, the datasets are balanced:
- **Goal**: Equal number of correct and false samples
- **Method**: Downsample majority class
- **Typical**: 100-150 samples per class per dataset

## Usage

### Training

```bash
python scripts/train_factoscope.py \
    --dataset_dir ./llm_factoscope-main/dataset_train \
    --max_samples 500
```

### Testing

```bash
python scripts/test_external_datasets.py \
    --dataset ./datasets/mmlu_10k_answers.json \
    --limit 100
```

### Batch Analysis

```bash
python scripts/batch_analysis.py \
    --dataset ./llm_factoscope-main/dataset_train/athlete_country_dataset.json \
    --limit 50
```

## Data Quality

### Good Questions

- Clear, unambiguous answers
- Factual (not opinions)
- Single correct answer
- Verifiable

Example: "What is the capital of France?" → "Paris"

### Avoid

- Multiple valid answers
- Subjective questions
- Time-dependent facts (without date)
- Ambiguous phrasing

## License and Attribution

### Original Factoscope Datasets

The training datasets are from the original Factoscope implementation. Please cite both the paper and repository:

```bibtex
@article{he2024factoscope,
  title={Factoscope: Uncovering LLMs' Factual Discernment through Inner States},
  author={He, Jinwen and Lyu, Yiqing and Ye, Yansong and Ho, Jennifer and others},
  year={2024}
}

@software{factoscope_original,
  author = {Ho, Jennifer and contributors},
  title = {LLM Factoscope},
  year = {2024},
  url = {https://github.com/JenniferHo97/llm_factoscope}
}
```

### External Datasets

External datasets (MMLU, HellaSwag, etc.) have their own licenses and citations. Please refer to their original sources for proper attribution.

### Extended Implementation

The dataset loading and conversion utilities are part of this extended implementation:

```bibtex
@software{factoscope_extended,
  author = {Mihai, Alexandru},
  title = {Factoscope: Extended Implementation with Modern LLM Support},
  year = {2026},
  url = {https://github.com/alexandru-mihai-94/llm_uncertainty_estimation},
  note = {Extended from original implementation by JenniferHo97}
}
```

## Creating High-Quality Datasets

### Best Practices

1. **Diversity**: Cover multiple knowledge domains
2. **Difficulty**: Mix easy and hard questions
3. **Balance**: Ensure model can get ~20-80% correct
4. **Verification**: Check answers are factually correct
5. **Format**: Consistent prompt structure

### Recommended Size

- **Minimum**: 100 questions per domain
- **Good**: 500 questions per domain
- **Optimal**: 1000+ questions per domain

### Domain Coverage

Include diverse topics:
- Geography (cities, countries, landmarks)
- History (events, people, dates)
- Arts (books, movies, music, paintings)
- Science (facts, discoveries, formulas)
- Sports (athletes, records, events)
- Culture (traditions, languages, cuisines)

## Support

For dataset-related questions:
- GitHub Issues: https://github.com/alexandru-mihai-94/llm_uncertainty_estimation/issues
- Check existing datasets for format examples
- See `examples/` for usage patterns
