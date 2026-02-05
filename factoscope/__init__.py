"""
Factoscope: LLM Uncertainty Estimation via k-NN in Embedding Space

ATTRIBUTION:
This library is built upon the original Factoscope implementation:
- Original Repository: https://github.com/JenniferHo97/llm_factoscope
- Original Authors: Jennifer Ho, Jinwen He, Yiqing Lyu, Yansong Ye, et al.
- Paper: "Factoscope: Uncovering LLMs' Factual Discernment through Inner States" (2024)

Core methodology and architecture extracted from original work.
Extended with modern LLM support, additional datasets, and modular structure.

Main components:
- FactDataCollector: Collect internal states from LLM
- FactDataPreprocessor: Normalize and balance data
- FactoscopeModel: Neural network for metric learning
- FactoscopeTrainer: Triplet loss training
- FactoscopeInference: Inference engine for predictions

See ATTRIBUTION.md for detailed credits.
"""

from .data_collection import FactDataCollector
from .preprocessing import FactDataPreprocessor
from .model import FactoscopeModel, HiddenStateEncoder, RankEncoder, ProbEncoder
from .training import FactoscopeTrainer, FactoscopeDataset
from .inference import FactoscopeInference

__version__ = '1.0.0'

__all__ = [
    'FactDataCollector',
    'FactDataPreprocessor',
    'FactoscopeModel',
    'HiddenStateEncoder',
    'RankEncoder',
    'ProbEncoder',
    'FactoscopeTrainer',
    'FactoscopeDataset',
    'FactoscopeInference',
]
