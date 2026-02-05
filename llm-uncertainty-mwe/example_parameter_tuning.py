"""
Example: Parameter tuning using MMLU dataset.

This script demonstrates how to:
1. Test different parameter combinations
2. Compare performance across settings
3. Find optimal feature weights
"""

import json
import numpy as np
from uncertainty_estimator import UncertaintyEstimator
from example_mmlu import load_dataset, evaluate_results, simple_answer_match


def test_parameter_combination(
    questions,
    true_answers,
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    token_weight=0.5,
    activation_weight=0.5,
    temperature=0.1,
):
    """
    Test a specific parameter combination.

    Args:
        questions: List of questions
        true_answers: List of correct answers
        model_name: Model to use
        token_weight: Weight for token features
        activation_weight: Weight for activation features
        temperature: Sampling temperature

    Returns:
        Dictionary of results and metrics
    """
    # Initialize estimator with parameters
    estimator = UncertaintyEstimator(
        model_name=model_name,
        token_weight=token_weight,
        activation_weight=activation_weight,
    )

    # Process questions
    results = estimator.estimate_batch(
        questions=questions,
        temperature=temperature,
        max_new_tokens=50,
        show_progress=False,
    )

    # Evaluate
    metrics = evaluate_results(results, true_answers, simple_answer_match)

    return {
        'params': {
            'token_weight': token_weight,
            'activation_weight': activation_weight,
            'temperature': temperature,
        },
        'metrics': metrics,
        'results': results,
    }


def main():
    print("="*70)
    print("Parameter Tuning Example")
    print("="*70)
    print()

    # Load sample dataset
    print("Loading dataset...")
    dataset = load_dataset('data/mmlu_sample_100.json')

    # Use smaller subset for parameter search
    n_samples = 15
    questions = [item['question'] for item in dataset[:n_samples]]
    true_answers = [item['answer'] for item in dataset[:n_samples]]

    print(f"Using {n_samples} questions for parameter tuning")
    print("(In practice, use more questions for robust tuning)\n")

    # Parameter grid
    print("="*70)
    print("TESTING PARAMETER COMBINATIONS")
    print("="*70)
    print()

    # Test different feature weight combinations
    weight_combinations = [
        (0.3, 0.7),  # More weight on activations
        (0.5, 0.5),  # Equal weight
        (0.7, 0.3),  # More weight on tokens
    ]

    # Test different temperatures
    temperatures = [0.1, 0.5, 1.0]

    all_results = []

    print("Testing weight combinations...")
    for token_w, activation_w in weight_combinations:
        print(f"\n  Token weight: {token_w:.1f}, Activation weight: {activation_w:.1f}")

        result = test_parameter_combination(
            questions=questions,
            true_answers=true_answers,
            token_weight=token_w,
            activation_weight=activation_w,
            temperature=0.1,
        )

        all_results.append(result)

        m = result['metrics']
        print(f"    Base accuracy: {m['base_accuracy']:.1%}")
        print(f"    Predictor accuracy: {m['predictor_accuracy']:.1%}")
        print(f"    F1 Score: {m['f1']:.2f}")

    print("\n" + "-"*70)
    print("\nTesting temperature settings...")
    for temp in temperatures:
        print(f"\n  Temperature: {temp:.1f}")

        result = test_parameter_combination(
            questions=questions,
            true_answers=true_answers,
            token_weight=0.5,
            activation_weight=0.5,
            temperature=temp,
        )

        all_results.append(result)

        m = result['metrics']
        print(f"    Base accuracy: {m['base_accuracy']:.1%}")
        print(f"    Predictor accuracy: {m['predictor_accuracy']:.1%}")
        print(f"    F1 Score: {m['f1']:.2f}")

    # Find best configuration
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS")
    print("="*70)

    # Sort by different metrics
    by_base_acc = sorted(all_results, key=lambda x: x['metrics']['base_accuracy'], reverse=True)
    by_pred_acc = sorted(all_results, key=lambda x: x['metrics']['predictor_accuracy'], reverse=True)
    by_f1 = sorted(all_results, key=lambda x: x['metrics']['f1'], reverse=True)

    print("\nBest base accuracy:")
    best = by_base_acc[0]
    print(f"  Parameters: {best['params']}")
    print(f"  Base accuracy: {best['metrics']['base_accuracy']:.1%}")
    print(f"  Predictor accuracy: {best['metrics']['predictor_accuracy']:.1%}")
    print(f"  F1: {best['metrics']['f1']:.2f}")

    print("\nBest predictor accuracy:")
    best = by_pred_acc[0]
    print(f"  Parameters: {best['params']}")
    print(f"  Base accuracy: {best['metrics']['base_accuracy']:.1%}")
    print(f"  Predictor accuracy: {best['metrics']['predictor_accuracy']:.1%}")
    print(f"  F1: {best['metrics']['f1']:.2f}")

    print("\nBest F1 score:")
    best = by_f1[0]
    print(f"  Parameters: {best['params']}")
    print(f"  Base accuracy: {best['metrics']['base_accuracy']:.1%}")
    print(f"  Predictor accuracy: {best['metrics']['predictor_accuracy']:.1%}")
    print(f"  F1: {best['metrics']['f1']:.2f}")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
Based on this parameter search:

1. Feature Weights:
   - Balanced weights (0.5/0.5) often work well as a starting point
   - Adjust based on your specific use case:
     * More token weight if you want fast, interpretable predictions
     * More activation weight if computational cost is not a concern

2. Temperature:
   - For small models, temperature has minimal impact on accuracy
   - Low temperature (0.1) gives more deterministic outputs
   - Higher temperature may help if you need diversity in responses

3. Next Steps:
   - Test on larger sample (100+ questions) for statistical significance
   - Use cross-validation to avoid overfitting to single split
   - Consider domain-specific tuning if your questions differ from MMLU
   - Track multiple metrics (accuracy, F1, calibration) not just one

4. Important Notes:
   - With <20% base accuracy, all configurations are limited
   - Consider using larger models for better base performance
   - Parameter tuning won't overcome fundamental model capacity issues

To run a more comprehensive grid search:
  - Increase n_samples to 100+
  - Test more granular weight combinations
  - Add layer selection to parameter grid
  - Test different model architectures
    """)


if __name__ == "__main__":
    main()
