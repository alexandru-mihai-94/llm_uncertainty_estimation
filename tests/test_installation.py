#!/usr/bin/env python3
"""
Test installation and verify all dependencies are available.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("\nTesting imports...")
    print("-" * 60)

    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'numpy': 'NumPy',
        'h5py': 'H5Py',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
    }

    failed = []

    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError as e:
            print(f"✗ {name} NOT installed: {e}")
            failed.append(name)

    return len(failed) == 0


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    print("-" * 60)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Devices: {torch.cuda.device_count()}")
            print(f"  Current: {torch.cuda.current_device()}")
            print(f"  Name: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available (CPU only)")
        return True
    except Exception as e:
        print(f"✗ Error testing CUDA: {e}")
        return False


def test_library():
    """Test that factoscope library can be imported"""
    print("\nTesting factoscope library...")
    print("-" * 60)

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from factoscope import (
            FactDataCollector,
            FactDataPreprocessor,
            FactoscopeModel,
            FactoscopeTrainer,
            FactoscopeInference
        )
        print("✓ All factoscope modules imported successfully")
        print("  - FactDataCollector")
        print("  - FactDataPreprocessor")
        print("  - FactoscopeModel")
        print("  - FactoscopeTrainer")
        print("  - FactoscopeInference")
        return True
    except ImportError as e:
        print(f"✗ Failed to import factoscope library: {e}")
        return False


def test_model_path():
    """Check if model path exists"""
    print("\nChecking model path...")
    print("-" * 60)

    model_path = Path('./models/Meta-Llama-3-8B')

    if model_path.exists():
        print(f"✓ Model directory exists: {model_path}")
        files = list(model_path.glob('*'))
        print(f"  Files found: {len(files)}")
        return True
    else:
        print(f"⚠ Model directory not found: {model_path}")
        print("  Download model first - see SETUP.md")
        return True  # Don't fail test, just warn


def main():
    print("\n" + "="*60)
    print("FACTOSCOPE INSTALLATION TEST")
    print("="*60)

    results = {
        'imports': test_imports(),
        'cuda': test_cuda(),
        'library': test_library(),
        'model': test_model_path(),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.upper()}: {status}")

    if all_passed:
        print("\n✓ Installation successful! You're ready to use Factoscope.")
        print("\nNext steps:")
        print("  1. Download model: See SETUP.md")
        print("  2. Run example: python examples/example_single_question.py")
        print("  3. Train model: python scripts/train_factoscope.py")
        return 0
    else:
        print("\n✗ Installation incomplete. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
