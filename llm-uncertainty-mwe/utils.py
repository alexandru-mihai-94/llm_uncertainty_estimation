"""
Utility functions for LLM uncertainty estimation.
"""

import torch
from typing import Dict, List, Optional


# Model-specific layer configurations
DEFAULT_LAYER_CONFIGS = {
    'llama-3.2-3b': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
    'llama32_3b': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
    'llama-3.1-8b': [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
    'llama31_8b': [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
    'phi-3': [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
    'phi3': [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
    'gemma-2-2b': [1, 3, 5, 7, 9, 11, 13, 15, 17],
    'gemma2_2b': [1, 3, 5, 7, 9, 11, 13, 15, 17],
    'default': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27],
}


# Common stopwords for content filtering
COMMON_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
    'to', 'of', 'and', 'in', 'that', 'it', 'for', 'on', 'with',
    'as', 'at', 'by', 'from', 'or', 'this', 'which', 'but', 'not',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'can', 'if', 'then'
}


def get_default_layers(model_name: str, num_layers: int) -> List[int]:
    """
    Get default target layers based on model name.

    Args:
        model_name: HuggingFace model identifier
        num_layers: Total number of layers in the model

    Returns:
        List of layer indices to sample
    """
    model_lower = model_name.lower()

    for key, layers in DEFAULT_LAYER_CONFIGS.items():
        if key in model_lower:
            valid_layers = [l for l in layers if l < num_layers]
            if valid_layers:
                return valid_layers

    # Fallback: sample ~10-14 layers evenly across the model
    n_target = min(14, num_layers)
    step = max(1, num_layers // n_target)
    return list(range(1, num_layers, step))[:n_target]


def format_prompt(question: str, model_name: str, tokenizer) -> str:
    """
    Format question for specific model architecture.

    Args:
        question: Question text
        model_name: Model identifier
        tokenizer: Model tokenizer

    Returns:
        Formatted prompt string
    """
    model_lower = model_name.lower()

    if "llama-3" in model_lower:
        messages = [
            {"role": "system", "content": "Answer questions directly and concisely."},
            {"role": "user", "content": f"{question}\n\nProvide a brief answer."}
        ]
        if hasattr(tokenizer, 'apply_chat_template'):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    elif "phi-3" in model_lower:
        return f"<|user|>\n{question}\nAnswer briefly.<|end|>\n<|assistant|>\n"
    elif "gemma" in model_lower:
        return f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"
    elif "qwen" in model_lower:
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

    return f"Question: {question}\nAnswer:"


def is_content_word(token_text: str) -> bool:
    """
    Check if token is a content word (not a stopword).

    Args:
        token_text: Token string

    Returns:
        True if content word, False otherwise
    """
    text = token_text.strip().lower()
    return len(text) > 3 and text.isalpha() and text not in COMMON_WORDS


def normalize_feature(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize feature to [0, 1] range.

    Args:
        value: Feature value
        min_val: Minimum expected value
        max_val: Maximum expected value

    Returns:
        Normalized value in [0, 1]
    """
    if max_val == min_val:
        return 0.5
    return min(1.0, max(0.0, (value - min_val) / (max_val - min_val)))


def compute_cosine_similarity(v1, v2):
    """
    Compute cosine similarity between two vectors.

    Args:
        v1: First vector (numpy array)
        v2: Second vector (numpy array)

    Returns:
        Cosine similarity in [-1, 1]
    """
    import numpy as np

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return 0.0

    return np.dot(v1, v2) / (n1 * n2)
