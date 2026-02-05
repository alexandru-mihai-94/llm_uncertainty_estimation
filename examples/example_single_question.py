#!/usr/bin/env python3
"""
Example: Single Question Inference

Demonstrates how to use Factoscope to predict confidence for a single question.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from factoscope import FactoscopeInference


def main():
    print("\n" + "="*80)
    print("FACTOSCOPE - SINGLE QUESTION EXAMPLE")
    print("="*80 + "\n")

    # Initialize inference engine
    print("Loading model (this may take a minute)...")

    engine = FactoscopeInference(
        model_path='./models/Meta-Llama-3-8B',
        factoscope_model_path='./factoscope_output/best_factoscope_model.pt',
        processed_data_path='./factoscope_output/processed_data.h5',
        device='cpu'  # Change to 'cuda' if GPU available
    )

    print("\n✓ Model loaded!\n")

    # Test questions
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
        "Who painted the Mona Lisa?",
        "What year did World War II end?"
    ]

    # Process each question
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}: {question}")
        print('-'*80)

        # Get prediction
        result = engine.predict_confidence(question)

        # Display results
        print(f"\n📝 Generated Answer: {result['answer']}")
        print(f"🎯 First Token: '{result['first_token']}'")
        print(f"\n📊 Confidence Metrics:")
        print(f"  Weighted Confidence: {result['confidence']:.2%}")
        print(f"  Simple Confidence:   {result['simple_confidence']:.2%}")
        print(f"  Prediction:          {result['prediction'].upper()}")

        print(f"\n📍 Nearest Neighbors:")
        print(f"  Distance to Correct: {result['nearest_correct_distance']:.4f}")
        print(f"  Distance to False:   {result['nearest_false_distance']:.4f}")
        print(f"  Distance Ratio:      {result['distance_ratio']:.4f}")

        if result['distance_ratio'] < 1:
            interpretation = "✓ Closer to correct examples (likely correct)"
        else:
            interpretation = "✗ Closer to false examples (likely incorrect)"
        print(f"  Interpretation:      {interpretation}")

        print(f"\n🔍 k-NN Breakdown (k=10):")
        print(f"  Correct neighbors: {result['top_k_neighbors']['correct']}")
        print(f"  False neighbors:   {result['top_k_neighbors']['false']}")

        print(f"\n📈 Token Statistics:")
        print(f"  Token Rank:        {result['rank']}")
        print(f"  Top Probability:   {result['top_prob']:.4f}")

        print('='*80)

    print("\n✓ Example complete!\n")


if __name__ == '__main__':
    main()
