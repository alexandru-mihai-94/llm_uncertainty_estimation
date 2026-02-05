#!/usr/bin/env python3
"""
Train Factoscope Model

Complete training pipeline:
1. Collect internal states from LLM
2. Preprocess and balance data
3. Train metric learning model with triplet loss
"""

import os
import sys
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import h5py
from torch.utils.data import random_split, DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from factoscope import (
    FactDataCollector,
    FactDataPreprocessor,
    FactoscopeModel,
    FactoscopeTrainer,
    FactoscopeDataset
)


def main():
    parser = argparse.ArgumentParser(description='Train Factoscope Model')

    # Model paths
    parser.add_argument('--model_path', type=str, default='./models/Meta-Llama-3-8B',
                       help='Path to LLM directory')
    parser.add_argument('--dataset_dir', type=str, default='./llm_factoscope-main/dataset_train',
                       help='Directory with dataset JSON files')
    parser.add_argument('--output_dir', type=str, default='./factoscope_output',
                       help='Output directory for features and models')

    # Data collection
    parser.add_argument('--max_samples', type=int, default=500,
                       help='Maximum samples per dataset')
    parser.add_argument('--skip_collection', action='store_true',
                       help='Skip data collection step (use existing features)')

    # Training
    parser.add_argument('--support_size', type=int, default=100,
                       help='Size of support set for evaluation')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training step')

    # Device
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print("\n" + "="*80)
    print("FACTOSCOPE TRAINING PIPELINE")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("="*80 + "\n")

    # ========================================================================
    # STEP 1: COLLECT DATA
    # ========================================================================
    dataset_dirs = []

    if not args.skip_collection:
        print("\n" + "="*80)
        print("STEP 1: COLLECTING INTERNAL STATES FROM LLM")
        print("="*80 + "\n")

        collector = FactDataCollector(args.model_path, device=args.device)

        # Find dataset files
        dataset_files = list(Path(args.dataset_dir).glob('*.json'))
        print(f"Found {len(dataset_files)} dataset files\n")

        # Process datasets
        for dataset_file in dataset_files[:5]:  # Limit to first 5 for demo
            dataset_name = dataset_file.stem.replace('_dataset', '')
            output_subdir = os.path.join(args.output_dir, 'features', dataset_name)

            print(f"\nProcessing: {dataset_name}")
            print("-" * 60)

            collector.process_dataset(
                str(dataset_file),
                output_subdir,
                max_samples=args.max_samples
            )
            dataset_dirs.append(output_subdir)
    else:
        # Use existing feature directories
        feature_dir = os.path.join(args.output_dir, 'features')
        dataset_dirs = [str(d) for d in Path(feature_dir).iterdir() if d.is_dir()]
        print(f"\nUsing {len(dataset_dirs)} existing feature directories")

    # ========================================================================
    # STEP 2: PREPROCESS DATA
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING DATA")
    print("="*80 + "\n")

    preprocessor = FactDataPreprocessor(args.output_dir)
    processed_file = os.path.join(args.output_dir, 'processed_data.h5')

    stats = preprocessor.prepare_training_data(dataset_dirs, processed_file)

    print(f"\n✓ Preprocessing complete!")
    print(f"  Total samples: {stats['num_samples']}")
    print(f"  Correct: {stats['num_correct']}")
    print(f"  False: {stats['num_false']}")

    # ========================================================================
    # STEP 3: TRAIN MODEL
    # ========================================================================
    if not args.skip_training:
        print("\n" + "="*80)
        print("STEP 3: TRAINING FACTOSCOPE MODEL")
        print("="*80 + "\n")

        # Load processed data
        with h5py.File(processed_file, 'r') as f:
            hidden_states = f['hidden_states'][:]
            ranks = f['ranks'][:]
            probs = f['topk_probs'][:]
            labels = f['labels'][:]
            num_layers = hidden_states.shape[1]
            hidden_dim = hidden_states.shape[2]

        print(f"Data loaded:")
        print(f"  Shape: {hidden_states.shape}")
        print(f"  Num layers: {num_layers}")
        print(f"  Hidden dim: {hidden_dim}\n")

        # Create dataset
        full_dataset = FactoscopeDataset(hidden_states, ranks, probs, labels)

        # Split into train/test/support
        total_size = len(full_dataset)
        train_size = int(0.7 * total_size)
        test_size = total_size - train_size - args.support_size

        train_dataset, test_dataset, support_dataset = random_split(
            full_dataset,
            [train_size, test_size, args.support_size]
        )

        print(f"Dataset splits:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Test: {len(test_dataset)}")
        print(f"  Support: {len(support_dataset)}\n")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        support_loader = DataLoader(support_dataset, batch_size=args.batch_size, shuffle=False)

        # Create model
        model = FactoscopeModel(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            emb_dim=32,
            final_dim=64
        )

        print(f"Model created:")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

        # Train
        trainer = FactoscopeTrainer(model, device=args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        best_accuracy = 0
        best_model_path = os.path.join(args.output_dir, 'best_factoscope_model.pt')

        print("Training...")
        print("-" * 80)

        for epoch in range(args.epochs):
            train_loss = trainer.train_epoch(train_loader, optimizer)

            if (epoch + 1) % 5 == 0:
                metrics = trainer.evaluate(test_loader, support_loader)
                print(f"\nEpoch {epoch+1}/{args.epochs}:")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Test Accuracy: {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1: {metrics['f1']:.4f}")

                if metrics['accuracy'] > best_accuracy:
                    best_accuracy = metrics['accuracy']
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  ✓ Saved best model (accuracy: {best_accuracy:.4f})")
            else:
                print(f"Epoch {epoch+1}/{args.epochs}: Loss = {train_loss:.4f}")

        # Final evaluation
        print("\n" + "="*80)
        print("FINAL EVALUATION")
        print("="*80 + "\n")

        model.load_state_dict(torch.load(best_model_path))
        final_metrics = trainer.evaluate(test_loader, support_loader)

        print("Final Results:")
        print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"  Precision: {final_metrics['precision']:.4f}")
        print(f"  Recall:    {final_metrics['recall']:.4f}")
        print(f"  F1 Score:  {final_metrics['f1']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {final_metrics['TP']:4d}  FP: {final_metrics['FP']:4d}")
        print(f"  FN: {final_metrics['FN']:4d}  TN: {final_metrics['TN']:4d}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nOutput saved to: {args.output_dir}")
    print(f"  - Features: {os.path.join(args.output_dir, 'features')}")
    print(f"  - Processed data: {processed_file}")
    if not args.skip_training:
        print(f"  - Best model: {best_model_path}")
        print(f"\nBest Accuracy: {best_accuracy:.4f}")
    print("\nNext steps:")
    print("  1. Test inference: python scripts/test_inference.py --interactive")
    print("  2. Batch analysis: python scripts/batch_analysis.py")
    print("  3. External testing: python scripts/test_external_datasets.py")
    print()


if __name__ == '__main__':
    main()
