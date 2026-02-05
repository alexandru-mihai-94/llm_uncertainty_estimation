# Repository Structure & GitHub Deployment Guide

This document explains the repository structure and how to deploy to GitHub.

## Attribution

**Important**: This repository is built upon the original Factoscope implementation:

- **Original Repository**: https://github.com/JenniferHo97/llm_factoscope
- **Original Authors**: Jennifer Ho, Jinwen He, Yiqing Lyu, Yansong Ye, and contributors
- **Paper**: "Factoscope: Uncovering LLMs' Factual Discernment through Inner States" (2024)

**What This Repository Adds**:
- Refactored modular architecture
- Support for modern LLMs (LLaMA-3, etc.)
- Extended dataset support (MMLU, HellaSwag, etc.)
- Comprehensive documentation
- Batch analysis and visualization tools
- Production-ready structure

## Repository Structure

```
factoscope-uncertainty/
│
├── README.md                          # Main documentation
├── SETUP.md                           # Installation guide
├── USAGE.md                           # Usage guide
├── REPOSITORY.md                      # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── factoscope/                        # Core library
│   ├── __init__.py                   # Package initialization
│   ├── data_collection.py            # Data collector (287 lines)
│   ├── preprocessing.py              # Data preprocessor (245 lines)
│   ├── model.py                      # Neural network models (189 lines)
│   ├── training.py                   # Training logic (226 lines)
│   └── inference.py                  # Inference engine (310 lines)
│
├── scripts/                           # Main scripts
│   ├── train_factoscope.py           # Complete training pipeline
│   ├── test_inference.py             # Interactive testing
│   ├── batch_analysis.py             # Batch analysis & plots
│   └── test_external_datasets.py     # External dataset testing
│
├── examples/                          # Usage examples
│   ├── example_single_question.py    # Single question demo
│   └── example_batch.py              # Batch processing demo
│
├── tests/                             # Tests
│   └── test_installation.py          # Installation verification
│
├── data/                              # Dataset information
│   └── README.md                      # Dataset documentation
│
└── docs/                              # Additional documentation
    └── (future: architecture diagrams, etc.)
```

## File Count

- **Python Files**: 14
- **Documentation**: 6 Markdown files
- **Total Lines of Code**: ~2,500 (excluding scripts)

## Library Components

### factoscope/data_collection.py

**Purpose**: Collect internal states from LLM

**Key Class**: `FactDataCollector`
- Loads LLM and tokenizer
- Generates responses and extracts hidden states
- Processes datasets and categorizes responses
- Saves to HDF5 format

### factoscope/preprocessing.py

**Purpose**: Normalize and prepare data for training

**Key Class**: `FactDataPreprocessor`
- Loads and balances data
- Z-score normalization
- Rank transformation
- Creates single training file

### factoscope/model.py

**Purpose**: Neural network architectures

**Key Classes**:
- `HiddenStateEncoder` - 1D CNN for hidden states
- `RankEncoder` - MLP for token ranks
- `ProbEncoder` - MLP for probabilities
- `FactoscopeModel` - Combined model

### factoscope/training.py

**Purpose**: Training with triplet loss

**Key Classes**:
- `FactoscopeDataset` - Triplet dataset
- `FactoscopeTrainer` - Training loop and evaluation

### factoscope/inference.py

**Purpose**: Inference engine

**Key Class**: `FactoscopeInference`
- Loads trained model
- Predicts confidence via k-NN
- Batch processing support

## Scripts Overview

### scripts/train_factoscope.py

Complete training pipeline:
1. Data collection from LLM
2. Preprocessing and balancing
3. Triplet learning
4. Model evaluation

Usage:
```bash
python scripts/train_factoscope.py --epochs 30
```

### scripts/test_inference.py

Interactive and batch testing:
- Interactive mode for manual testing
- Batch mode for multiple questions
- JSON output support

Usage:
```bash
python scripts/test_inference.py --interactive
```

### scripts/batch_analysis.py

Batch evaluation with visualizations:
- Processes multiple questions
- Generates 7 types of plots
- Summary statistics

Usage:
```bash
python scripts/batch_analysis.py --dataset data.json
```

### scripts/test_external_datasets.py

Test on external datasets (MMLU, HellaSwag, etc.):
- Loads model once
- Tests multiple datasets
- Generates plots per dataset
- Summary report

Usage:
```bash
python scripts/test_external_datasets.py --test_all
```

## Deploying to GitHub

### Prerequisites

1. GitHub account: alexandru-mihai-94
2. Repository created: llm_uncertainty_estimation
3. SSH key configured

### Step 1: Initialize Git

```bash
cd /Users/alexandrumihai/Documents/llm_uncertainty_estimation/factoscope-uncertainty

git init
git add .
git commit -m "Initial commit: Factoscope uncertainty estimation framework"
```

### Step 2: Add Remote

```bash
git remote add origin git@github.com:alexandru-mihai-94/llm_uncertainty_estimation.git
```

### Step 3: Push to GitHub

```bash
# Create and push to new branch
git checkout -b factoscope-framework
git push -u origin factoscope-framework

# Or push to main
git branch -M main
git push -u origin main
```

### Step 4: Verify

Visit: https://github.com/alexandru-mihai-94/llm_uncertainty_estimation

## Repository Best Practices

### What to Commit

✅ **Do commit**:
- All Python source code
- Documentation (*.md)
- Configuration files (requirements.txt, .gitignore)
- Example scripts
- Tests

❌ **Don't commit**:
- Model files (*.pt, *.h5) - too large
- Generated data (factoscope_output/)
- Virtual environments (venv/)
- Dataset files (datasets/)
- Results and plots

### Large Files

For models and datasets, use:

1. **Git LFS** (Large File Storage)
   ```bash
   git lfs install
   git lfs track "*.pt"
   git lfs track "*.h5"
   git add .gitattributes
   ```

2. **External hosting**
   - Hugging Face Hub
   - Zenodo
   - Google Drive
   - Link in README

3. **Download script**
   - Provide script to download models
   - Include in SETUP.md

## Branch Strategy

### Recommended Branches

- `main` - Stable, tested code
- `develop` - Active development
- `feature/xyz` - New features
- `bugfix/xyz` - Bug fixes

### Workflow

```bash
# Create feature branch
git checkout -b feature/new-model

# Make changes
git add .
git commit -m "Add new model architecture"

# Push
git push origin feature/new-model

# Create pull request on GitHub
```

## Documentation Updates

When adding new features:

1. Update relevant *.md files
2. Add docstrings to new functions
3. Update examples if API changes
4. Add to USAGE.md if user-facing

## Testing

Before pushing:

```bash
# Test installation
python tests/test_installation.py

# Test imports
python -c "from factoscope import *; print('✓ All imports successful')"

# Test example
python examples/example_single_question.py
```

## Continuous Integration (Future)

Consider adding:

- `.github/workflows/tests.yml` - Run tests on push
- `.github/workflows/lint.yml` - Code quality checks
- Pre-commit hooks for formatting

Example `.github/workflows/tests.yml`:
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python tests/test_installation.py
```

## Release Process

### Version Numbering

Use Semantic Versioning: MAJOR.MINOR.PATCH

- `1.0.0` - Initial release
- `1.1.0` - New features (backward compatible)
- `1.0.1` - Bug fixes
- `2.0.0` - Breaking changes

### Creating a Release

```bash
# Tag version
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Create release on GitHub
# Go to: https://github.com/alexandru-mihai-94/llm_uncertainty_estimation/releases
# Click "Draft a new release"
```

## Maintenance

### Regular Tasks

- Update dependencies in requirements.txt
- Test with latest PyTorch/Transformers versions
- Update documentation
- Review and merge pull requests
- Respond to issues

### Security

- Don't commit API keys or tokens
- Use `.env` files for secrets (git-ignored)
- Keep dependencies updated

## Support & Community

### Getting Help

- GitHub Issues for bugs
- Discussions for questions
- Pull requests for contributions

### Contributing

Welcome contributions:
- Bug fixes
- New features
- Documentation improvements
- Example scripts
- Test coverage

## Next Steps

After deployment:

1. ✅ Push to GitHub
2. ✅ Add README badges (build status, license, etc.)
3. ✅ Create GitHub Pages documentation
4. ✅ Add to PyPI (optional)
5. ✅ Write blog post or paper

## Links

- Repository: https://github.com/alexandru-mihai-94/llm_uncertainty_estimation
- Issues: https://github.com/alexandru-mihai-94/llm_uncertainty_estimation/issues
- Wiki: https://github.com/alexandru-mihai-94/llm_uncertainty_estimation/wiki

## License

MIT License - See LICENSE file

Open source, free to use, modify, and distribute.
