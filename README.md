# LLM Uncertainty Estimation: Two Complementary Approaches

Comprehensive framework for estimating uncertainty in Large Language Models using two distinct methodologies: token-level statistics for small models and k-NN in embedding space for larger models.

## Attribution

**This repository builds upon the original Factoscope implementation:**

- **Original Repository**: [JenniferHo97/llm_factoscope](https://github.com/JenniferHo97/llm_factoscope/tree/main)
- **Original Authors**: Jennifer Ho, Jinwen He, Yiqing Lyu, Yansong Ye, and contributors
- **Original Paper**: "Factoscope: Uncovering LLMs' Factual Discernment through Inner States" (He et al., 2024)

### What Was Modified from Original

**Extracted and adapted from original repository:**
- Core data collection functions (updated for Transformers 4.30+)
- Hidden state extraction logic (modernized for newer APIs)
- Triplet learning architecture (refactored into modules)
- k-NN inference mechanism (extracted and optimized)
- Training datasets (from original Factoscope)

**Newly added functionality in this extended version:**
- Support for newer LLaMA models (Meta-Llama-3-8B, etc.)
- Compatibility with modern Transformers library (4.30+)
- **Extended dataset support** (MMLU, HellaSwag, CosmosQA, etc.)
- **Multiple dataset formats** and automatic conversion
- **Batch analysis** with 7 types of visualizations
- **External dataset testing** pipeline
- **Small model experiments** (3-4B parameters)
- **Token-level statistics approach** for resource-constrained scenarios
- **Modular library structure** for reusability
- **Comprehensive documentation** (7 guides + examples)

**See [ATTRIBUTION.md](ATTRIBUTION.md) for detailed breakdown of original vs. new code.**

## Repository Structure

This repository contains two complementary approaches to LLM uncertainty estimation:

### 1. Factoscope Approach (Large Models, 8B+ parameters)

**Location**: `factoscope/` library and `scripts/`

**Method**: k-Nearest Neighbors in learned embedding space using triplet loss

**Best for**:
- Large models (8B+ parameters)
- High accuracy requirements
- Research applications
- When computational resources are available

**Performance**: 65-75% accuracy in distinguishing correct vs incorrect answers on factual QA

### 2. Token-Statistics Approach (Small Models, 3-4B parameters)

**Location**: `llm-uncertainty-mwe/`

**Method**: Direct analysis of token probabilities and layer activation patterns

**Best for**:
- Small models (3-4B parameters)
- Resource-constrained environments
- Quick deployment
- Exploratory analysis

**Key Finding**: Small models (Llama 3.2 3B, Phi-3 Mini) achieved <20% accuracy on MMLU (below 25% random baseline). Predictors trained on this data are biased toward "incorrect" due to extreme class imbalance (~85% incorrect examples).

## Quick Start

### Factoscope Approach (Large Models)

```python
from factoscope import FactoscopeInference

# Initialize
engine = FactoscopeInference(
    model_path='./models/Meta-Llama-3-8B',
    factoscope_model_path='./factoscope_output/best_factoscope_model.pt',
    processed_data_path='./factoscope_output/processed_data.h5'
)

# Predict
result = engine.predict_confidence("What is the capital of France?")

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Prediction: {result['prediction']}")  # 'correct' or 'false'
```

### Token-Statistics Approach (Small Models)

```python
from llm_uncertainty_mwe.uncertainty_estimator import UncertaintyEstimator

# Initialize
estimator = UncertaintyEstimator(
    model_name="meta-llama/Llama-3.2-3B-Instruct"
)

# Estimate
result = estimator.estimate(
    question="What is the capital of France?"
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence_score']:.2%}")
print(f"Predicted Correct: {result['predicted_correct']}")
```

## Installation

```bash
# Clone repository
git clone git@github.com:alexandru-mihai-94/llm_uncertainty_estimation.git
cd llm_uncertainty_estimation/factoscope-uncertainty

# Install dependencies for Factoscope approach
pip install -r requirements.txt

# Install dependencies for small model approach
cd llm-uncertainty-mwe
pip install -r requirements.txt
```

## Project Structure

```
factoscope-uncertainty/
│
├── README.md                          # This file
├── ATTRIBUTION.md                     # Detailed attribution info
├── SETUP.md                           # Installation guide
├── USAGE.md                           # Usage guide
├── requirements.txt                   # Python dependencies
│
├── factoscope/                        # Factoscope library (large models)
│   ├── __init__.py
│   ├── data_collection.py            # LLM state collection
│   ├── preprocessing.py              # Data normalization
│   ├── model.py                      # Neural network architectures
│   ├── training.py                   # Triplet learning trainer
│   └── inference.py                  # Inference engine
│
├── scripts/                           # Factoscope scripts
│   ├── train_factoscope.py           # Train on factual datasets
│   ├── test_inference.py             # Interactive testing
│   ├── batch_analysis.py             # Batch evaluation & plots
│   └── test_external_datasets.py     # MMLU, HellaSwag, etc.
│
├── examples/                          # Factoscope examples
│   ├── example_single_question.py
│   └── example_batch.py
│
├── llm-uncertainty-mwe/               # Small model approach (NEW)
│   ├── README.md                      # Small model documentation
│   ├── uncertainty_estimator.py       # Main estimator class
│   ├── feature_extractor.py           # Feature extraction
│   ├── example_single.py              # Single question example
│   ├── example_batch.py               # Batch processing
│   ├── example_mmlu.py                # MMLU evaluation
│   └── data/
│       ├── mmlu_10k_answers.json      # Full MMLU (10K questions)
│       └── mmlu_sample_100.json       # Sample (100 questions)
│
└── data/                              # Dataset information
    └── README.md                      # Dataset sources
```

## Experiments and Results

### Small Model Experiments (3-4B Parameters)

**Models Tested**: Llama 3.2 3B Instruct, Phi-3 Mini

**Dataset**: MMLU (Massive Multitask Language Understanding) - 10,000 questions

**Key Findings**:

1. **Base Model Performance**
   - Llama 3.2 3B: ~15-18% accuracy on MMLU
   - Phi-3 Mini: ~16-19% accuracy on MMLU
   - Both significantly below random chance (25% for 4-choice questions)

2. **Temperature Independence**
   - Minimal accuracy variation across temperature settings (0.1 to 1.0)
   - Variation <3% across all temperature values
   - Likely an artifact of floor performance - limited room for further degradation

3. **Predictor Performance**
   - ROC AUC: ~0.85 (but contextualized by extreme class imbalance)
   - Precision for "correct": ~0.15 (reflects base accuracy)
   - Predictors predominantly predict "incorrect" due to ~85% negative examples

4. **Feature Analysis**
   - **Token-level features**: Confidence, margins, ranking patterns
   - **Activation features**: Layer norms, cross-layer consistency, trajectory
   - Combined approach: 40% token features, 60% activation features
   - Features showed discriminative ability but limited by class imbalance

**Conclusion**: Small models (3-4B) struggle with knowledge-intensive tasks like MMLU. While uncertainty estimation is technically feasible, the extreme class imbalance and low base accuracy limit practical utility. For production applications, larger models (8B+) with the Factoscope approach are recommended.

### Large Model Experiments (8B Parameters)

**Model**: Meta-Llama-3-8B

**Dataset**: Factoscope training datasets (athlete-country, book-author, movie-director, etc.)

**Results**:

1. **Classification Accuracy**: 65-75% in distinguishing correct vs incorrect answers
2. **Calibration**: Confidence scores correlate with actual correctness
3. **Generalization**: Works on unseen question types (MMLU, external datasets)
4. **k-NN Performance**: k=10 neighbors provides optimal balance

**Conclusion**: The Factoscope approach with larger models provides reliable uncertainty estimates suitable for production use cases like answer validation and active learning.

## Methodology Comparison

| Aspect | Small Models (Token Stats) | Large Models (Factoscope) |
|--------|---------------------------|--------------------------|
| **Model Size** | 3-4B parameters | 8B+ parameters |
| **Approach** | Direct feature analysis | k-NN in embedding space |
| **Training** | Feature weight tuning | Triplet loss learning |
| **Accuracy** | Limited by base model | 65-75% discrimination |
| **Computational Cost** | 2x inference time | Model loading + k-NN |
| **Use Case** | Exploration, research | Production applications |
| **Data Required** | Minimal | Requires training set |

## Training Workflows

### Factoscope Training (Large Models)

```bash
python scripts/train_factoscope.py \
    --model_path ./models/Meta-Llama-3-8B \
    --dataset_dir ./llm_factoscope-main/dataset_train \
    --output_dir ./factoscope_output \
    --max_samples 500 \
    --epochs 30
```

**Outputs**:
- `factoscope_output/best_factoscope_model.pt` (trained model)
- `factoscope_output/processed_data.h5` (preprocessed data)
- `factoscope_output/features/` (per-dataset features)

### Token-Statistics Calibration (Small Models)

The small model approach uses empirically-derived feature weights and does not require separate training. Simply initialize and use:

```python
from llm_uncertainty_mwe.uncertainty_estimator import UncertaintyEstimator

estimator = UncertaintyEstimator(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    token_weight=0.4,
    activation_weight=0.6
)
```

## Inference and Testing

### Interactive Mode (Both Approaches)

**Factoscope**:
```bash
python scripts/test_inference.py --interactive
```

**Small Models**:
```bash
cd llm-uncertainty-mwe
python example_single.py
```

### Batch Analysis

**Factoscope**:
```bash
python scripts/batch_analysis.py \
    --dataset ./data/test_questions.json \
    --plot_dir ./plots
```

**Small Models**:
```bash
cd llm-uncertainty-mwe
python example_batch.py
```

### External Dataset Testing (Factoscope)

```bash
python scripts/test_external_datasets.py \
    --dataset ./datasets/mmlu_10k_answers.json \
    --limit 100
```

## Use Cases

### 1. Production Answer Validation (Factoscope)

Use the large model approach for high-stakes applications:

- Flag low-confidence answers for human review
- Automated quality control in QA systems
- Confidence-weighted answer ranking

### 2. Resource-Constrained Scenarios (Small Models)

Use token-statistics approach when:

- Limited GPU resources
- Quick deployment needed
- Exploratory analysis
- Educational purposes

### 3. Research Applications (Both)

Compare approaches to understand:

- How model size affects uncertainty estimation
- Trade-offs between methods
- Feature importance across scales

## Performance Metrics

### Factoscope (Meta-Llama-3-8B)

- **Accuracy**: 65-75% (correct vs incorrect classification)
- **Precision**: 0.70-0.75
- **Recall**: 0.65-0.72
- **F1 Score**: 0.68-0.73
- **Inference Time**: ~500ms per question (CPU), ~150ms (GPU)

### Token Statistics (Llama 3.2 3B)

- **Base Accuracy**: 15-20% on MMLU
- **ROC AUC**: ~0.85 (with extreme class imbalance caveat)
- **Precision**: ~0.15 for "correct" class
- **Inference Time**: ~200ms per question (CPU), ~50ms (GPU)
- **Temperature Independence**: <3% variation (0.1 to 1.0)

## Citation

If you use this code in research, please cite the original Factoscope paper and acknowledge this extended implementation:

### Original Factoscope

```bibtex
@article{he2024factoscope,
  title={Factoscope: Uncovering LLMs' Factual Discernment through Inner States},
  author={He, Jinwen and Lyu, Yiqing and Ye, Yansong and Ho, Jennifer and others},
  journal={arXiv preprint},
  year={2024}
}
```

### Original Implementation

```bibtex
@software{factoscope_original,
  author = {Ho, Jennifer and contributors},
  title = {LLM Factoscope},
  year = {2024},
  url = {https://github.com/JenniferHo97/llm_factoscope}
}
```

### This Extended Implementation

```bibtex
@software{factoscope_extended,
  author = {Mihai, Alexandru},
  title = {LLM Uncertainty Estimation: Factoscope Extension and Small Model Experiments},
  year = {2026},
  url = {https://github.com/alexandru-mihai-94/llm_uncertainty_estimation},
  note = {Extended from original Factoscope implementation with small model experiments},
  institution = {OIST}
}
```

## Requirements

### Factoscope (Large Models)

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Meta-Llama-3-8B model
- 16GB+ RAM (32GB recommended)
- GPU with 16GB+ VRAM (for training)

### Token Statistics (Small Models)

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Llama 3.2 3B or Phi-3 Mini
- 8GB+ RAM
- GPU with 8GB+ VRAM (optional)

## Documentation

- [ATTRIBUTION.md](ATTRIBUTION.md) - Detailed attribution and credits
- [SETUP.md](SETUP.md) - Installation and model download
- [USAGE.md](USAGE.md) - Detailed usage guide
- [data/README.md](data/README.md) - Dataset information
- [llm-uncertainty-mwe/README.md](llm-uncertainty-mwe/README.md) - Small model approach
- [llm-uncertainty-mwe/DATASET_USAGE.md](llm-uncertainty-mwe/DATASET_USAGE.md) - MMLU dataset guide

## Credits

### Original Implementation

This work builds upon the foundation provided by:
- **Original Authors**: Jennifer Ho, Jinwen He, Yiqing Lyu, Yansong Ye, and contributors
- **Original Repository**: https://github.com/JenniferHo97/llm_factoscope
- **Paper**: Factoscope: Uncovering LLMs' Factual Discernment through Inner States

### Modifications and Extensions

- **Extended by**: Alexandru Mihai (2026)
- **Institution**: OIST
- **Key additions**:
  - Modern model support (LLaMA-3, Transformers 4.30+)
  - Extended datasets (MMLU, HellaSwag, CosmosQA)
  - Small model experiments and analysis
  - Token-statistics approach
  - Modular architecture
  - Comprehensive documentation

## License

MIT License - See LICENSE file for details

**Note**: This repository maintains the same permissive licensing spirit as the original Factoscope implementation while adding proper attribution.

## Contributing

Contributions welcome! Please open an issue or pull request.

When contributing, please:
- Maintain compatibility with core methodologies
- Add tests for new features
- Update documentation accordingly
- Cite relevant papers and implementations
- Clearly distinguish between approaches (small vs large models)

## Support

For questions or issues:
- **Original Factoscope methodology**: See [original repository](https://github.com/JenniferHo97/llm_factoscope) issues
- **Extended features and small model experiments**: GitHub Issues on this repository
- **Documentation**: See USAGE.md and inline docstrings
