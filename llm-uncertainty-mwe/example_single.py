"""
Example: Single question uncertainty estimation.

This script demonstrates basic usage of the uncertainty estimator
on a single question.
"""

from uncertainty_estimator import UncertaintyEstimator


def main():
    # Initialize estimator
    print("="*60)
    print("Single Question Example")
    print("="*60)
    print()

    estimator = UncertaintyEstimator(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        token_weight=0.5,
        activation_weight=0.5,
    )

    # Ask a question
    question = "What is the capital of France?"

    print(f"\nQuestion: {question}")
    print("\nGenerating answer and estimating confidence...")

    result = estimator.estimate(
        question=question,
        temperature=0.1,
        max_new_tokens=50
    )

    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Answer: {result['answer']}")
    print(f"Confidence Score: {result['confidence_score']:.2%}")
    print(f"Uncertainty Score: {result['uncertainty_score']:.2%}")
    print(f"Predicted Correct: {result['predicted_correct']}")
    print(f"Number of Tokens: {result['num_tokens']}")

    print(f"\nUncertainty Breakdown:")
    print(f"  Token-based: {result['token_uncertainty']:.3f}")
    print(f"  Activation-based: {result['activation_uncertainty']:.3f}")

    print(f"\nKey Token Features:")
    token_feats = result['features']['token']
    print(f"  Mean confidence: {token_feats['mean_confidence']:.3f}")
    print(f"  Mean margin: {token_feats['mean_margin']:.3f}")
    print(f"  Min margin: {token_feats['min_margin']:.3f}")

    print(f"\nKey Activation Features:")
    act_feats = result['features']['activation']
    print(f"  Mean norm: {act_feats['mean_norm']:.1f}")
    print(f"  Cross-layer consistency: {act_feats['mean_cross_layer_consistency']:.3f}")
    print(f"  Layer agreement: {act_feats['layer_agreement']:.3f}")

    print()
    print("NOTE: For small models (<4B parameters), predictions are often biased")
    print("toward 'incorrect' due to low base accuracy (~15-20%) on complex questions.")


if __name__ == "__main__":
    main()
