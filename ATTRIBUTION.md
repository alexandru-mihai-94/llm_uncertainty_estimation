# Attribution and Credits

## Original Work

This repository is built upon the **original Factoscope implementation**:

### Original Repository
- **URL**: https://github.com/JenniferHo97/llm_factoscope
- **Authors**: Jennifer Ho, Jinwen He, Yiqing Lyu, Yansong Ye, and contributors
- **License**: [Check original repository]
- **Year**: 2024

### Original Paper
```bibtex
@article{he2024factoscope,
  title={Factoscope: Uncovering LLMs' Factual Discernment through Inner States},
  author={He, Jinwen and Lyu, Yiqing and Ye, Yansong and Ho, Jennifer and others},
  journal={arXiv preprint},
  year={2024}
}
```

## What Was Extracted from Original

The following components were **extracted and adapted** from the original Factoscope repository:

### Core Methodology (from original)
1. **Hidden State Collection**
   - Extracting internal states from LLM layers
   - Token probability and rank tracking
   - Response categorization logic

2. **Triplet Learning Architecture**
   - Hidden state encoder (1D CNN)
   - Rank encoder (MLP)
   - Probability encoder (MLP)
   - Triplet loss training

3. **k-NN Inference**
   - Support set construction
   - k-nearest neighbor classification
   - Distance-based confidence calculation

4. **Training Datasets**
   - Factual question datasets (athlete_country, book_author, etc.)
   - Dataset structure and format
   - Original data collection methodology

### Modifications to Original Code

**Updated for compatibility**:
- Modern Transformers API (4.30+) compatibility
- PyTorch 2.0+ support
- Updated model loading for newer LLaMA versions
- Fixed deprecated function calls

**Refactored for modularity**:
- Extracted monolithic scripts into library modules
- Separated concerns (data collection, preprocessing, training, inference)
- Added proper Python package structure
- Improved error handling and validation

## Newly Added Functionality

The following features are **NEW** and not part of the original implementation:

### 1. Extended Model Support
- ✨ Meta-Llama-3-8B compatibility
- ✨ Automatic device detection and optimization
- ✨ Support for newer model architectures

### 2. Extended Dataset Support
- ✨ MMLU dataset integration (10,000+ questions)
- ✨ HellaSwag support
- ✨ CosmosQA support
- ✨ TruthfulQA support
- ✨ Automatic dataset format conversion
- ✨ Custom dataset creation utilities

### 3. Batch Analysis Tools
- ✨ 7 types of visualization plots
- ✨ Comprehensive statistical analysis
- ✨ Confidence distribution analysis
- ✨ Nearest neighbor breakdown
- ✨ Interactive plotting

### 4. Production Features
- ✨ Modular library structure (factoscope package)
- ✨ Comprehensive documentation (7 guides)
- ✨ Example scripts and tutorials
- ✨ Installation tests
- ✨ Error handling and logging
- ✨ Type hints and docstrings

### 5. Testing Infrastructure
- ✨ External dataset testing pipeline
- ✨ Batch processing capabilities
- ✨ Interactive testing mode
- ✨ Results visualization and export

### 6. Documentation
- ✨ README with quick start
- ✨ Detailed setup guide
- ✨ Comprehensive usage documentation
- ✨ Dataset creation guide
- ✨ Repository deployment guide
- ✨ API reference documentation

## Code Attribution Breakdown

### Original Code (Adapted)
**File**: `factoscope/data_collection.py`
- **Original source**: Data collection logic from original repo
- **Modifications**: Updated for Transformers 4.30+, added error handling
- **Lines**: ~287 (60% original logic, 40% new)

**File**: `factoscope/model.py`
- **Original source**: Neural network architectures from original repo
- **Modifications**: Refactored into separate classes, added type hints
- **Lines**: ~189 (80% original architecture, 20% new structure)

**File**: `factoscope/training.py`
- **Original source**: Triplet training from original repo
- **Modifications**: Separated into classes, improved evaluation
- **Lines**: ~226 (70% original training, 30% new features)

**File**: `factoscope/preprocessing.py`
- **Original source**: Data preprocessing from original repo
- **Modifications**: Extracted into module, added statistics
- **Lines**: ~245 (65% original preprocessing, 35% new)

**File**: `factoscope/inference.py`
- **Original source**: k-NN inference from original repo
- **Modifications**: Added batch support, improved confidence metrics
- **Lines**: ~310 (60% original inference, 40% new)

### Entirely New Code
**Files**:
- `scripts/batch_analysis.py` - 100% new (visualization and analysis)
- `scripts/test_external_datasets.py` - 100% new (external testing)
- `examples/*.py` - 100% new (demonstrations)
- `tests/*.py` - 100% new (testing infrastructure)
- All documentation files - 100% new

## Credits

### Original Factoscope Team
- **Primary Authors**: Jennifer Ho, Jinwen He, Yiqing Lyu, Yansong Ye
- **Contribution**: Core methodology, original implementation, training datasets
- **Repository**: https://github.com/JenniferHo97/llm_factoscope

### Extended Implementation
- **Developer**: Alexandru Mihai
- **Year**: 2026
- **Contribution**: Refactoring, modern LLM support, extended datasets, documentation
- **Repository**: https://github.com/alexandru-mihai-94/llm_uncertainty_estimation

## Citation Guidelines

### For Research Using This Code

**If using the core Factoscope methodology**, cite the original paper:
```bibtex
@article{he2024factoscope,
  title={Factoscope: Uncovering LLMs' Factual Discernment through Inner States},
  author={He, Jinwen and Lyu, Yiqing and Ye, Yansong and Ho, Jennifer and others},
  year={2024}
}
```

**If using the extended features** (MMLU support, batch analysis, etc.), additionally cite:
```bibtex
@software{factoscope_extended,
  author = {Mihai, Alexandru},
  title = {Factoscope: Extended Implementation with Modern LLM Support},
  year = {2026},
  url = {https://github.com/alexandru-mihai-94/llm_uncertainty_estimation},
  note = {Extended from original implementation by JenniferHo97}
}
```

## License

This extended implementation maintains compatibility with the original repository's licensing:

- **Original code**: Licensed under original Factoscope repository license
- **New code**: MIT License (see LICENSE file)
- **Combined work**: Respects both licenses

## Acknowledgments

Special thanks to:
- **Jennifer Ho** and the original Factoscope team for the foundational work
- **Jinwen He, Yiqing Lyu, Yansong Ye** for the research and methodology
- All contributors to the original repository

This work would not be possible without their excellent research and open-source contribution.

## Contact

**For questions about**:
- **Original methodology**: See original repository issues
- **Extended features**: Open issue in this repository
- **Collaboration**: Contact Alexandru Mihai via GitHub

## Disclaimer

This is an independent extension and refactoring of the original Factoscope implementation. It is not officially affiliated with or endorsed by the original authors, though it respectfully builds upon their work with proper attribution.

---

**Last Updated**: February 2026
