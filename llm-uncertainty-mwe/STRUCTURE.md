# Repository Structure

```
llm-uncertainty-mwe/
├── README.md                    # Main documentation with overview and quick start
├── SETUP.md                     # Detailed setup and troubleshooting guide
├── STRUCTURE.md                 # This file - repository structure guide
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore patterns
│
├── Core Library Files:
├── utils.py                     # Utility functions (layer configs, formatting, etc.)
├── feature_extractor.py         # Token and activation feature extraction
├── uncertainty_estimator.py     # Main estimator class
│
├── Example Scripts:
├── test_installation.py         # Verify installation and dependencies
├── example_single.py            # Single question example
├── example_batch.py             # Batch processing example
├── example_dataset.py           # Dataset evaluation with ground truth
├── example_mmlu.py              # MMLU dataset evaluation
├── example_parameter_tuning.py  # Parameter optimization example
│
└── data/                        # Included datasets
    ├── README.md                # Dataset documentation
    ├── mmlu_10k_answers.json    # Full MMLU dataset (10,000 questions)
    └── mmlu_sample_100.json     # Sample dataset (100 questions)
```

## File Descriptions

### Core Library

**utils.py** (4.2 KB)
- Model-specific layer configurations
- Prompt formatting for different architectures
- Helper functions (content word detection, normalization, cosine similarity)
- Constants (stopword lists, default layer maps)

**feature_extractor.py** (9.4 KB)
- `FeatureExtractor` class
- Token-level feature extraction (confidence, margins, rankings, temporal patterns)
- Activation-based feature extraction (norms, consistency, sparsity, trajectory)
- Feature aggregation across layers

**uncertainty_estimator.py** (12.0 KB)
- `UncertaintyEstimator` main class
- Model loading and initialization
- Answer generation with hidden state extraction
- Feature combination and uncertainty scoring
- Single and batch processing methods

### Examples

**test_installation.py** (4.5 KB)
- Tests package imports
- Verifies local module imports
- Checks CUDA availability
- Tests basic functionality
- Provides helpful error messages

**example_single.py** (2.1 KB)
- Demonstrates single question processing
- Shows how to initialize estimator
- Displays detailed results and feature breakdown
- Minimal example for quick testing

**example_batch.py** (3.2 KB)
- Batch processing of multiple questions
- Aggregate statistics computation
- Confidence distribution analysis
- Demonstrates efficiency for multiple queries

**example_dataset.py** (5.9 KB)
- Dataset evaluation with ground truth
- Automatic answer matching
- Metrics computation (accuracy, precision, recall, F1)
- Calibration analysis (ECE)
- Confusion matrix generation

### Documentation

**README.md** (5.8 KB)
- Project overview and key findings
- Quick start guide
- Usage examples
- Feature descriptions
- Performance metrics
- Citation information

**SETUP.md** (4.8 KB)
- Step-by-step installation instructions
- Memory requirements by model
- Troubleshooting common issues
- Configuration options
- Using different models

## Usage Workflow

### 1. Installation
```bash
pip install -r requirements.txt
python3 test_installation.py
```

### 2. Basic Usage
```python
from uncertainty_estimator import UncertaintyEstimator

estimator = UncertaintyEstimator(
    model_name="meta-llama/Llama-3.2-3B-Instruct"
)

result = estimator.estimate("What is 2+2?")
print(f"Confidence: {result['confidence_score']:.2%}")
```

### 3. Batch Processing
```python
questions = ["Q1?", "Q2?", "Q3?"]
results = estimator.estimate_batch(questions)
```

### 4. Dataset Evaluation
```python
# Load your dataset
dataset = [{"question": "...", "correct_answer": "..."}]

# Evaluate
results = estimator.estimate_batch([d['question'] for d in dataset])

# Compute metrics (see example_dataset.py for full code)
```

## Customization Points

### 1. Feature Weights
Edit `TOKEN_WEIGHTS` and `ACTIVATION_WEIGHTS` in `uncertainty_estimator.py`

### 2. Layer Selection
Modify `DEFAULT_LAYER_CONFIGS` in `utils.py` or pass `target_layers` parameter

### 3. Prompt Formatting
Update `format_prompt()` in `utils.py` for custom model templates

### 4. Feature Extraction
Extend `FeatureExtractor` class to add new features

### 5. Uncertainty Computation
Modify `_compute_token_uncertainty()` and `_compute_activation_uncertainty()` methods

## Key Design Principles

1. **Modular**: Separate concerns (utils, feature extraction, estimation)
2. **Reusable**: Easy to import and use in other projects
3. **Extensible**: Clear structure for adding features or models
4. **Documented**: Comprehensive docstrings and examples
5. **Tested**: Installation verification script

## Dependencies

- **torch**: Model inference and tensor operations
- **transformers**: HuggingFace model loading
- **numpy**: Numerical computing and feature computation
- **tqdm** (optional): Progress bars
- **matplotlib** (optional): Visualization
- **scikit-learn** (optional): Evaluation metrics

## Next Steps

1. Run `python3 test_installation.py` to verify setup
2. Try examples to understand the API
3. Customize for your use case
4. Integrate into your project

## Contributing

To add new features:
1. Add helper functions to `utils.py`
2. Extend `FeatureExtractor` for new feature types
3. Update `UncertaintyEstimator` to use new features
4. Add examples demonstrating new functionality
5. Update documentation
