"""
Test script to verify all imports used in examples are available.
This doesn't require torch/transformers to be installed.
"""

import sys

def test_imports():
    """Test all imports used in examples"""

    errors = []

    # Test 1: Basic imports
    print("Testing basic imports...")
    try:
        from funcroute import FuncRoute, TrainingConfig
        print("  ✅ FuncRoute, TrainingConfig")
    except ImportError as e:
        errors.append(f"❌ FuncRoute imports: {e}")

    # Test 2: Core config
    print("\nTesting core config...")
    try:
        from funcroute.core.config import ToolDefinition
        print("  ✅ ToolDefinition")
    except ImportError as e:
        errors.append(f"❌ ToolDefinition: {e}")

    # Test 3: Data pipeline
    print("\nTesting data pipeline...")
    try:
        from funcroute.data.generator import SyntheticDataGenerator
        print("  ✅ SyntheticDataGenerator")
    except ImportError as e:
        errors.append(f"❌ SyntheticDataGenerator: {e}")

    try:
        from funcroute.data.splitter import PatternGroupSplitter
        print("  ✅ PatternGroupSplitter")
    except ImportError as e:
        errors.append(f"❌ PatternGroupSplitter: {e}")

    try:
        from funcroute.data.validator import DataValidator
        print("  ✅ DataValidator")
    except ImportError as e:
        errors.append(f"❌ DataValidator: {e}")

    # Test 4: Evaluation
    print("\nTesting evaluation...")
    try:
        from funcroute.evaluation import Evaluator, Visualizer
        print("  ✅ Evaluator, Visualizer")
    except ImportError as e:
        errors.append(f"❌ Evaluation: {e}")

    # Test 5: Inference
    print("\nTesting inference...")
    try:
        from funcroute.inference import Predictor, RouteCache, WarmupCache
        print("  ✅ Predictor, RouteCache, WarmupCache")
    except ImportError as e:
        errors.append(f"❌ Inference: {e}")

    # Test 6: Server (optional - requires FastAPI)
    print("\nTesting server (optional)...")
    try:
        from funcroute.inference.server import create_app, run_server
        print("  ✅ create_app, run_server (FastAPI available)")
    except ImportError as e:
        print(f"  ⚠️  Server: {e}")
        print("     (Install with: pip install 'funcroute[server]')")

    # Report results
    print("\n" + "=" * 80)
    if errors:
        print("❌ IMPORT TEST FAILED\n")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("✅ ALL IMPORTS SUCCESSFUL")
        print("\nAll example imports are available!")
        print("Note: torch/transformers not required for import test")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
