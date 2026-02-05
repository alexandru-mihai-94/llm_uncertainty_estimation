#!/usr/bin/env python3
"""
Test Factoscope on External Datasets (MMLU, HellaSwag, CosmosQA, etc.)
Optimized: Loads model once and reuses for all testing
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import argparse

from factoscope_inference import FactoscopeInference
from factoscope_batch_analysis import FactoscopeAnalyzer


def load_mmlu_dataset(dataset_path: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Load MMLU dataset

    Returns:
        List of dicts with 'prompt', 'answer', 'id' keys
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Convert MMLU format to standard format
    # MMLU has: {"id": X, "question": "...", "answer": "..."}
    processed_data = []
    for item in data[:limit] if limit else data:
        # Create a prompt that the model can complete
        question = item['question'].strip()

        # Format as a completion task
        prompt = f"Question: {question}\nAnswer:"

        processed_data.append({
            'id': item['id'],
            'prompt': prompt,
            'expected_answer': item['answer'],
            'question': question
        })

    return processed_data


def load_generic_dataset(dataset_path: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Load a generic dataset with flexible format

    Handles various formats:
    - {"question": "...", "answer": "..."}
    - {"prompt": "...", "answer": "..."}
    - {"text": "...", "label": "..."}
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    for i, item in enumerate(data[:limit] if limit else data):
        # Determine prompt field
        if 'prompt' in item:
            prompt = item['prompt']
        elif 'question' in item:
            prompt = f"Question: {item['question']}\nAnswer:"
        elif 'text' in item:
            prompt = item['text']
        else:
            print(f"Warning: Could not find prompt/question/text in item {i}")
            continue

        # Determine answer field
        if 'answer' in item:
            answer = item['answer']
        elif 'label' in item:
            answer = item['label']
        elif 'expected_answer' in item:
            answer = item['expected_answer']
        else:
            answer = None

        processed_data.append({
            'id': item.get('id', i),
            'prompt': prompt,
            'expected_answer': answer,
            'original': item
        })

    return processed_data


def evaluate_answer(generated: str, expected: str) -> bool:
    """
    Evaluate if generated answer matches expected answer
    More lenient matching for diverse answer formats
    """
    # Normalize both answers
    gen_norm = generated.lower().strip()
    exp_norm = expected.lower().strip()

    # Exact match
    if gen_norm == exp_norm:
        return True

    # Check if expected answer is contained in generated (for short answers)
    if exp_norm in gen_norm:
        return True

    # Check if first word matches (for single-word answers)
    gen_first = gen_norm.split()[0] if gen_norm.split() else ""
    exp_first = exp_norm.split()[0] if exp_norm.split() else ""
    if gen_first == exp_first and len(exp_first) > 2:
        return True

    return False


def test_on_dataset(
    engine: FactoscopeInference,
    dataset_name: str,
    dataset_path: str,
    limit: Optional[int] = None,
    output_dir: Path = Path("./external_test_results")
) -> Dict:
    """
    Test Factoscope on a single external dataset

    Args:
        engine: Loaded FactoscopeInference engine
        dataset_name: Name of the dataset
        dataset_path: Path to dataset file
        limit: Maximum number of questions to test
        output_dir: Directory to save results

    Returns:
        Dictionary with results and statistics
    """
    print(f"\n{'='*80}")
    print(f"Testing on: {dataset_name}")
    print(f"{'='*80}\n")

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")

    # Determine loader based on filename
    if 'mmlu' in dataset_name.lower():
        data = load_mmlu_dataset(dataset_path, limit=limit)
    else:
        data = load_generic_dataset(dataset_path, limit=limit)

    print(f"✓ Loaded {len(data)} questions\n")

    # Run inference
    print("Running inference...")
    results = []

    for i, item in enumerate(data):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(data)}")

        # Get prediction
        result = engine.predict_confidence(item['prompt'])

        # Evaluate correctness if expected answer is available
        if item.get('expected_answer'):
            is_correct = evaluate_answer(result['first_token'], item['expected_answer'])
            result['expected_answer'] = item['expected_answer']
            result['is_correct'] = is_correct

            # Override prediction based on actual correctness
            # (since the k-NN might not know the true label)
            result['actual_correctness'] = 'correct' if is_correct else 'false'

        result['question_id'] = item['id']
        if 'question' in item:
            result['question'] = item['question']

        results.append(result)

    print(f"✓ Completed inference on {len(results)} questions\n")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / f"{dataset_name}_results.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {results_file}\n")

    # Calculate statistics
    stats = calculate_statistics(results, dataset_name)

    # Generate plots
    plot_dir = output_dir / f"{dataset_name}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    analyzer = FactoscopeAnalyzer(results)
    analyzer.print_summary_statistics()
    analyzer.plot_all(output_dir=str(plot_dir))

    print(f"✓ Plots saved to: {plot_dir}\n")

    return {
        'dataset_name': dataset_name,
        'results_file': str(results_file),
        'plot_dir': str(plot_dir),
        'statistics': stats,
        'num_questions': len(results)
    }


def calculate_statistics(results: List[Dict], dataset_name: str) -> Dict:
    """Calculate detailed statistics for a dataset"""
    import numpy as np

    stats = {
        'total_questions': len(results),
        'mean_confidence': np.mean([r['confidence'] for r in results]),
        'std_confidence': np.std([r['confidence'] for r in results]),
        'median_confidence': np.median([r['confidence'] for r in results]),
        'mean_top_prob': np.mean([r['top_prob'] for r in results]),
    }

    # If we have correctness information
    if 'is_correct' in results[0]:
        correct = [r for r in results if r.get('is_correct', False)]
        incorrect = [r for r in results if not r.get('is_correct', True)]

        stats['accuracy'] = len(correct) / len(results) * 100
        stats['num_correct'] = len(correct)
        stats['num_incorrect'] = len(incorrect)

        if correct:
            stats['mean_confidence_correct'] = np.mean([r['confidence'] for r in correct])
        if incorrect:
            stats['mean_confidence_incorrect'] = np.mean([r['confidence'] for r in incorrect])

        # Correlation between factoscope prediction and actual correctness
        factoscope_correct = sum(1 for r in results if r.get('prediction') == 'correct')
        actual_correct = sum(1 for r in results if r.get('is_correct', False))
        agreement = sum(1 for r in results
                       if (r.get('prediction') == 'correct') == r.get('is_correct', False))

        stats['factoscope_accuracy'] = factoscope_correct / len(results) * 100
        stats['actual_accuracy'] = actual_correct / len(results) * 100
        stats['prediction_agreement'] = agreement / len(results) * 100

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Test Factoscope on External Datasets (MMLU, etc.)'
    )

    # Model paths
    parser.add_argument('--model_path', type=str, default='./models/Meta-Llama-3-8B',
                       help='Path to LLM')
    parser.add_argument('--factoscope_model', type=str, default='./best_factoscope_model.pt',
                       help='Path to trained factoscope model')
    parser.add_argument('--processed_data', type=str, default='./processed_data.h5',
                       help='Path to processed training data')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu or cuda)')

    # Dataset selection
    parser.add_argument('--dataset', type=str,
                       help='Path to specific dataset file')
    parser.add_argument('--dataset_dir', type=str, default='./datasets',
                       help='Directory containing datasets')
    parser.add_argument('--test_all', action='store_true',
                       help='Test on all datasets in dataset_dir')
    parser.add_argument('--limit', type=int,
                       help='Limit number of questions per dataset')

    # Output
    parser.add_argument('--output_dir', type=str, default='./external_test_results',
                       help='Output directory for results')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("FACTOSCOPE EXTERNAL DATASET TESTING")
    print("="*80)
    print(f"Results will be saved to: {run_dir}")
    print("="*80 + "\n")

    # Determine which datasets to test
    datasets_to_test = []

    if args.dataset:
        # Single dataset
        datasets_to_test.append({
            'name': Path(args.dataset).stem,
            'path': args.dataset
        })
    elif args.test_all:
        # All datasets in directory
        dataset_dir = Path(args.dataset_dir)
        if dataset_dir.exists():
            for ds_file in sorted(dataset_dir.glob("*.json")):
                datasets_to_test.append({
                    'name': ds_file.stem,
                    'path': str(ds_file)
                })
        else:
            print(f"Error: Dataset directory not found: {dataset_dir}")
            return
    else:
        # Interactive selection
        dataset_dir = Path(args.dataset_dir)
        if dataset_dir.exists():
            available = sorted(dataset_dir.glob("*.json"))
            print("Available datasets:\n")
            for i, ds in enumerate(available, 1):
                size_mb = ds.stat().st_size / (1024 * 1024)
                print(f"  {i}. {ds.name} ({size_mb:.1f} MB)")

            print("\nSelect dataset number (or 'all' to test all): ", end='')
            choice = input().strip()

            if choice.lower() == 'all':
                datasets_to_test = [{'name': ds.stem, 'path': str(ds)} for ds in available]
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(available):
                        datasets_to_test.append({
                            'name': available[idx].stem,
                            'path': str(available[idx])
                        })
                    else:
                        print("Invalid selection")
                        return
                except ValueError:
                    print("Invalid input")
                    return
        else:
            print(f"Error: Dataset directory not found: {dataset_dir}")
            return

    if not datasets_to_test:
        print("No datasets to test")
        return

    print(f"\nWill test on {len(datasets_to_test)} dataset(s):")
    for ds in datasets_to_test:
        print(f"  - {ds['name']}")
    print()

    # ========================================================================
    # LOAD MODEL ONCE
    # ========================================================================
    print("INITIALIZING MODEL (loading once for all datasets)...")
    print("-"*80)

    try:
        engine = FactoscopeInference(
            model_path=args.model_path,
            factoscope_model_path=args.factoscope_model,
            processed_data_path=args.processed_data,
            device=args.device
        )
        print("\n✓ Model loaded successfully!")
    except Exception as e:
        print(f"\n✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*80 + "\n")

    # ========================================================================
    # TEST ON EACH DATASET
    # ========================================================================
    all_results = {}
    successful = []
    failed = []

    for i, dataset_info in enumerate(datasets_to_test, 1):
        print(f"\n[{i}/{len(datasets_to_test)}] Testing on: {dataset_info['name']}")
        print("-"*80)

        try:
            result = test_on_dataset(
                engine=engine,
                dataset_name=dataset_info['name'],
                dataset_path=dataset_info['path'],
                limit=args.limit,
                output_dir=run_dir
            )

            all_results[dataset_info['name']] = result
            successful.append(dataset_info['name'])

            print(f"✓ Success: {dataset_info['name']}\n")

        except Exception as e:
            print(f"✗ Error testing {dataset_info['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed.append(dataset_info['name'])

    # ========================================================================
    # GENERATE SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print(f"\nTotal datasets tested: {len(datasets_to_test)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\n✓ Successfully tested:")
        for name in successful:
            stats = all_results[name]['statistics']
            print(f"\n  {name}:")
            print(f"    Questions: {stats['total_questions']}")
            if 'accuracy' in stats:
                print(f"    Actual Accuracy: {stats['actual_accuracy']:.2f}%")
                print(f"    Factoscope Agreement: {stats['prediction_agreement']:.2f}%")
            print(f"    Mean Confidence: {stats['mean_confidence']:.3f}")

    if failed:
        print(f"\n✗ Failed:")
        for name in failed:
            print(f"  - {name}")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'datasets_tested': len(datasets_to_test),
        'successful': successful,
        'failed': failed,
        'results': all_results,
        'configuration': {
            'limit_per_dataset': args.limit,
            'device': args.device,
            'model_path': args.model_path
        }
    }

    summary_file = run_dir / "test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_file}")
    print(f"✓ All results saved to: {run_dir}")
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
