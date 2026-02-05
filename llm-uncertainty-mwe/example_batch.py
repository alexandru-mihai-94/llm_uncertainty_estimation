"""
Example: Batch processing with uncertainty estimation.

Demonstrates processing multiple questions and analyzing
confidence distributions.
"""

from uncertainty_estimator import UncertaintyEstimator
import numpy as np


def main():
    print("="*60)
    print("Batch Processing Example")
    print("="*60)
    print()

    # Initialize estimator
    estimator = UncertaintyEstimator(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        token_weight=0.5,
        activation_weight=0.5,
    )

    # Multiple questions with varying difficulty
    questions = [
        "What is 2+2?",
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light in vacuum?",
        "Who was the first president of the United States?",
        "What is the chemical formula for water?",
        "In what year did World War II end?",
        "What is the largest planet in our solar system?",
    ]

    print(f"Processing {len(questions)} questions...")
    print()

    # Process batch
    results = estimator.estimate_batch(
        questions=questions,
        temperature=0.1,
        max_new_tokens=50,
        show_progress=True
    )

    # Display individual results
    print("\n" + "="*60)
    print("INDIVIDUAL RESULTS")
    print("="*60)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Q: {result['question']}")
        print(f"   A: {result['answer']}")
        print(f"   Confidence: {result['confidence_score']:.2%} | "
              f"Predicted: {'✓ Correct' if result['predicted_correct'] else '✗ Incorrect'}")

    # Aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE STATISTICS")
    print("="*60)

    confidence_scores = [r['confidence_score'] for r in results]
    uncertainty_scores = [r['uncertainty_score'] for r in results]
    predicted_correct = [r['predicted_correct'] for r in results]

    print(f"\nConfidence Statistics:")
    print(f"  Mean: {np.mean(confidence_scores):.2%}")
    print(f"  Std: {np.std(confidence_scores):.2%}")
    print(f"  Min: {np.min(confidence_scores):.2%}")
    print(f"  Max: {np.max(confidence_scores):.2%}")

    print(f"\nPredictions:")
    print(f"  Predicted Correct: {sum(predicted_correct)}/{len(predicted_correct)} "
          f"({100*sum(predicted_correct)/len(predicted_correct):.1f}%)")
    print(f"  Predicted Incorrect: {len(predicted_correct) - sum(predicted_correct)}/{len(predicted_correct)} "
          f"({100*(len(predicted_correct) - sum(predicted_correct))/len(predicted_correct):.1f}%)")

    # Confidence bins
    high_conf = sum(1 for c in confidence_scores if c >= 0.65)
    med_conf = sum(1 for c in confidence_scores if 0.40 <= c < 0.65)
    low_conf = sum(1 for c in confidence_scores if c < 0.40)

    print(f"\nConfidence Distribution:")
    print(f"  High (≥65%): {high_conf} questions")
    print(f"  Medium (40-65%): {med_conf} questions")
    print(f"  Low (<40%): {low_conf} questions")

    print("\n" + "="*60)
    print("NOTE: Small models often show bias toward 'incorrect'")
    print("predictions due to low base accuracy on complex questions.")
    print("="*60)


if __name__ == "__main__":
    main()
