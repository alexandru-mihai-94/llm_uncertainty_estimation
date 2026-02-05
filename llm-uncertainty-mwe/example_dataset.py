"""
Example: Dataset evaluation with ground truth.

Demonstrates evaluating the estimator on a dataset with known answers
and computing calibration metrics.
"""

from uncertainty_estimator import UncertaintyEstimator
import numpy as np
from typing import List, Dict


def evaluate_predictions(results: List[Dict], ground_truth: List[bool]) -> Dict:
    """
    Evaluate prediction performance against ground truth.

    Args:
        results: List of estimation results
        ground_truth: List of boolean correctness labels

    Returns:
        Dictionary of evaluation metrics
    """
    assert len(results) == len(ground_truth), "Mismatched lengths"

    predictions = [r['predicted_correct'] for r in results]
    confidences = [r['confidence_score'] for r in results]

    # Confusion matrix
    tp = sum(1 for p, gt in zip(predictions, ground_truth) if p and gt)
    fp = sum(1 for p, gt in zip(predictions, ground_truth) if p and not gt)
    tn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and not gt)
    fn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and gt)

    # Metrics
    accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Calibration: group by confidence bins
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_accuracies = []
    bin_confidences = []

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        mask = [(low <= c < high) for c in confidences]
        if sum(mask) > 0:
            bin_acc = np.mean([gt for gt, m in zip(ground_truth, mask) if m])
            bin_conf = np.mean([c for c, m in zip(confidences, mask) if m])
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)

    # Expected Calibration Error (ECE)
    ece = 0.0
    if bin_accuracies:
        ece = np.mean([abs(acc - conf) for acc, conf in zip(bin_accuracies, bin_confidences)])

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ece': ece,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn},
        'num_correct': sum(ground_truth),
        'num_incorrect': len(ground_truth) - sum(ground_truth),
    }


def main():
    print("="*60)
    print("Dataset Evaluation Example")
    print("="*60)
    print()

    # Initialize estimator
    estimator = UncertaintyEstimator(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        token_weight=0.5,
        activation_weight=0.5,
    )

    # Example dataset (questions with ground truth correctness)
    # In practice, you would load this from a file
    dataset = [
        {"question": "What is 2+2?", "correct_answer": "4"},
        {"question": "What is the capital of France?", "correct_answer": "Paris"},
        {"question": "Who wrote Romeo and Juliet?", "correct_answer": "Shakespeare"},
        {"question": "What is the speed of light?", "correct_answer": "299,792,458 m/s"},
        {"question": "Who was the first US president?", "correct_answer": "George Washington"},
    ]

    print(f"Evaluating on {len(dataset)} questions...")
    print()

    # Process questions
    questions = [d['question'] for d in dataset]
    results = estimator.estimate_batch(
        questions=questions,
        temperature=0.1,
        max_new_tokens=50,
        show_progress=True
    )

    # Manually evaluate correctness (in practice, use automated matching)
    print("\nPlease manually verify if answers are correct:")
    ground_truth = []

    for i, (result, data) in enumerate(zip(results, dataset), 1):
        print(f"\n{i}. Q: {data['question']}")
        print(f"   Model answer: {result['answer']}")
        print(f"   Expected: {data['correct_answer']}")

        # For this example, we'll do simple string matching (not robust!)
        # In practice, use more sophisticated answer matching
        is_correct = data['correct_answer'].lower() in result['answer'].lower()
        ground_truth.append(is_correct)

        print(f"   Evaluation: {'✓ Correct' if is_correct else '✗ Incorrect'}")
        print(f"   Model predicted: {'✓ Correct' if result['predicted_correct'] else '✗ Incorrect'} "
              f"(confidence: {result['confidence_score']:.2%})")

    # Compute metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)

    metrics = evaluate_predictions(results, ground_truth)

    print(f"\nGround Truth Distribution:")
    print(f"  Actually Correct: {metrics['num_correct']}/{len(ground_truth)}")
    print(f"  Actually Incorrect: {metrics['num_incorrect']}/{len(ground_truth)}")

    print(f"\nPrediction Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")
    print(f"  F1 Score: {metrics['f1']:.2%}")

    print(f"\nCalibration:")
    print(f"  Expected Calibration Error (ECE): {metrics['ece']:.3f}")

    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (predicted correct, was correct): {cm['tp']}")
    print(f"  False Positives (predicted correct, was incorrect): {cm['fp']}")
    print(f"  True Negatives (predicted incorrect, was incorrect): {cm['tn']}")
    print(f"  False Negatives (predicted incorrect, was correct): {cm['fn']}")

    print("\n" + "="*60)
    print("NOTE: This is a minimal example with only 5 questions.")
    print("For robust evaluation, use larger datasets (100+ questions).")
    print("Small models typically show high bias toward 'incorrect'")
    print("predictions due to low base accuracy.")
    print("="*60)


if __name__ == "__main__":
    main()
