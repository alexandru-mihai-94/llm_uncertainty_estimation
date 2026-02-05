"""
Test script to verify installation and basic functionality.

This script checks that all dependencies are installed correctly
without loading large models.
"""

import sys


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
        return False

    try:
        import numpy
        print(f"✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False

    try:
        import tqdm
        print(f"✓ tqdm {tqdm.__version__}")
    except ImportError:
        print("⚠ tqdm not installed (optional, for progress bars)")

    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print("⚠ Matplotlib not installed (optional, for plotting)")

    try:
        import sklearn
        print(f"✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        print("⚠ scikit-learn not installed (optional, for metrics)")

    return True


def test_local_imports():
    """Test that local modules can be imported."""
    print("\nTesting local modules...")

    try:
        import utils
        print("✓ utils.py")
    except ImportError as e:
        print(f"✗ utils.py import failed: {e}")
        return False

    try:
        import feature_extractor
        print("✓ feature_extractor.py")
    except ImportError as e:
        print(f"✗ feature_extractor.py import failed: {e}")
        return False

    try:
        import uncertainty_estimator
        print("✓ uncertainty_estimator.py")
    except ImportError as e:
        print(f"✗ uncertainty_estimator.py import failed: {e}")
        return False

    return True


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("⚠ CUDA not available (will use CPU)")
            print("  This is fine, but inference will be slower")

    except Exception as e:
        print(f"⚠ Error checking CUDA: {e}")

    return True


def test_basic_functionality():
    """Test basic functionality without loading models."""
    print("\nTesting basic functionality...")

    try:
        from utils import get_default_layers, is_content_word, normalize_feature

        # Test get_default_layers
        layers = get_default_layers("llama-3.2-3b", num_layers=28)
        assert isinstance(layers, list)
        assert len(layers) > 0
        print(f"✓ get_default_layers: {len(layers)} layers")

        # Test is_content_word
        assert is_content_word("important") == True
        assert is_content_word("the") == False
        print("✓ is_content_word")

        # Test normalize_feature
        val = normalize_feature(0.5, 0.0, 1.0)
        assert 0.0 <= val <= 1.0
        print("✓ normalize_feature")

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

    return True


def print_system_info():
    """Print system information."""
    print("\nSystem Information:")
    print(f"  Python version: {sys.version}")
    print(f"  Platform: {sys.platform}")

    try:
        import torch
        print(f"  PyTorch backend: {torch.version.cuda if torch.cuda.is_available() else 'CPU'}")
    except:
        pass


def main():
    """Run all tests."""
    print("="*60)
    print("LLM Uncertainty Estimation - Installation Test")
    print("="*60)
    print()

    # Print system info
    print_system_info()
    print()

    # Run tests
    tests = [
        ("Package imports", test_imports),
        ("Local module imports", test_local_imports),
        ("CUDA availability", test_cuda),
        ("Basic functionality", test_basic_functionality),
    ]

    results = []
    for name, test_func in tests:
        print("-"*60)
        result = test_func()
        results.append((name, result))
        print()

    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False

    print()

    if all_passed:
        print("="*60)
        print("✓ All tests passed!")
        print("="*60)
        print()
        print("You're ready to run the examples:")
        print("  python3 example_single.py")
        print("  python3 example_batch.py")
        print("  python3 example_dataset.py")
        print()
        print("Note: First run will download the model (~6GB)")
        print("="*60)
        return 0
    else:
        print("="*60)
        print("✗ Some tests failed")
        print("="*60)
        print()
        print("Please check the error messages above and:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check that you're in the correct directory")
        print("3. Verify Python version is 3.8+")
        print()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
