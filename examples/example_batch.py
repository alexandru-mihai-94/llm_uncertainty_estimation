#!/usr/bin/env python3
"""
Example: Batch Processing

Demonstrates how to process multiple questions and analyze results.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from factoscope import FactoscopeInference


def main():
    print("\n" + "="*80)
    print("FACTOSCOPE - BATCH PROCESSING EXAMPLE")
    print("="*80 + "\n")

    # Sample dataset
    test_questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
        "Who painted the Mona Lisa?",
        "What year did World War II end?",
        "What is the speed of light?",
        "Who invented the telephone?",
        "What is the chemical symbol for gold?",
        "Who was the first president of the United States?",
        "What is the tallest mountain in the world?"
    ]

    # Initialize engine
    print("Loading model...")
    engine = FactoscopeInference(
        model_path='./models/Meta-Llama-3-8B',
        factoscope_model_path='./factoscope_output/best_factoscope_model.pt',
        processed_data_path='./factoscope_output/processed_data.h5',
        device='cpu'
    )
    print("✓ Model loaded!\n")

    # Batch processing
    print(f"Processing {len(test_questions)} questions...")
    print("-" * 80 + "\n")

    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"[{i}/{len(test_questions)}] {question[:50]}...")
        result = engine.predict_confidence(question)
        results.append(result)

    # Summary statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80 + "\n")

    total = len(results)
    predicted_correct = sum(1 for r in results if r['prediction'] == 'correct')
    predicted_false = total - predicted_correct

    avg_confidence = sum(r['confidence'] for r in results) / total
    high_conf = sum(1 for r in results if r['confidence'] > 0.7)
    low_conf = sum(1 for r in results if r['confidence'] < 0.3)

    print(f"Total Questions: {total}")
    print(f"\nPredictions:")
    print(f"  Predicted Correct: {predicted_correct} ({predicted_correct/total*100:.1f}%)")
    print(f"  Predicted False:   {predicted_false} ({predicted_false/total*100:.1f}%)")
    print(f"\nConfidence:")
    print(f"  Average:      {avg_confidence:.3f}")
    print(f"  High (>0.7):  {high_conf} ({high_conf/total*100:.1f}%)")
    print(f"  Low (<0.3):   {low_conf} ({low_conf/total*100:.1f}%)")

    # Top 5 most confident
    print(f"\n{'='*80}")
    print("TOP 5 MOST CONFIDENT ANSWERS")
    print('='*80 + "\n")

    sorted_by_conf = sorted(results, key=lambda x: x['confidence'], reverse=True)
    for i, result in enumerate(sorted_by_conf[:5], 1):
        print(f"{i}. {result['prompt'][:60]}...")
        print(f"   Answer: {result['answer'][:40]}")
        print(f"   Confidence: {result['confidence']:.3f} | Prediction: {result['prediction']}")
        print()

    # Top 5 least confident
    print(f"{'='*80}")
    print("TOP 5 LEAST CONFIDENT ANSWERS")
    print('='*80 + "\n")

    for i, result in enumerate(sorted_by_conf[-5:], 1):
        print(f"{i}. {result['prompt'][:60]}...")
        print(f"   Answer: {result['answer'][:40]}")
        print(f"   Confidence: {result['confidence']:.3f} | Prediction: {result['prediction']}")
        print()

    # Save results
    output_file = 'example_batch_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Full results saved to: {output_file}")
    print("\n✓ Example complete!\n")


if __name__ == '__main__':
    main()
