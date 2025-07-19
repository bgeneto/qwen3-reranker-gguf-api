#!/usr/bin/env python3
"""
Debug script to diagnose model loading issues.
Run this to get detailed information about the model file and llama-cpp-python compatibility.
"""

import os
import sys
import logging
import hashlib
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.config import settings
from app.model_downloader import verify_gguf_file, calculate_file_hash
from app.logger import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)


def check_system_info():
    """Check system information."""
    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")

    try:
        import llama_cpp

        print(f"llama-cpp-python version: {llama_cpp.__version__}")
    except ImportError as e:
        print(f"llama-cpp-python not found: {e}")
        return False
    except AttributeError:
        print("llama-cpp-python version info not available")

    return True


def check_model_file():
    """Check the model file."""
    print("\n=== Model File Information ===")
    model_path = settings.model_path
    print(f"Model path: {model_path}")
    print(f"Model filename: {settings.MODEL_FILENAME}")
    print(f"Model link: {settings.MODEL_LINK}")

    if not os.path.exists(model_path):
        print("❌ Model file does not exist")
        return False

    print("✅ Model file exists")

    # File size
    file_size = os.path.getsize(model_path)
    print(f"File size: {file_size:,} bytes ({file_size / (1024**3):.2f} GB)")

    # Permissions
    readable = os.access(model_path, os.R_OK)
    print(f"Readable: {'✅' if readable else '❌'}")

    if not readable:
        print("❌ Cannot read model file - permission issue")
        return False

    # GGUF verification
    is_valid_gguf = verify_gguf_file(model_path)
    print(f"Valid GGUF format: {'✅' if is_valid_gguf else '❌'}")

    if not is_valid_gguf:
        print("❌ Model file is not in valid GGUF format")
        return False

    # File hash
    print("Calculating file hash...")
    file_hash = calculate_file_hash(model_path)
    print(f"SHA256 hash: {file_hash}")

    return True


def test_model_loading():
    """Test model loading with different configurations."""
    print("\n=== Model Loading Tests ===")

    try:
        from llama_cpp import Llama
    except ImportError as e:
        print(f"❌ Cannot import Llama: {e}")
        return False

    model_path = settings.model_path

    # Test configurations
    test_configs = [
        {
            "name": "Minimal CPU-only",
            "config": {
                "n_gpu_layers": 0,
                "n_ctx": 512,
                "n_batch": 8,
                "n_threads": 1,
                "verbose": True,
            },
        },
        {
            "name": "Small CPU-only",
            "config": {
                "n_gpu_layers": 0,
                "n_ctx": 2048,
                "n_batch": 128,
                "n_threads": 2,
                "verbose": True,
            },
        },
        {
            "name": "Default settings",
            "config": {
                "n_gpu_layers": settings.N_GPU_LAYERS,
                "n_ctx": settings.N_CTX,
                "n_batch": settings.N_BATCH,
                "n_threads": settings.N_THREADS,
                "verbose": True,
            },
        },
    ]

    for test in test_configs:
        print(f"\nTesting: {test['name']}")
        print(f"Config: {test['config']}")

        try:
            llm = Llama(model_path=model_path, logits_all=True, **test["config"])
            print(f"✅ {test['name']} - Model loaded successfully!")

            # Test a simple tokenization
            tokens = llm.tokenize(b"test")
            print(f"✅ Tokenization test passed (tokens: {len(tokens)})")

            # Clean up
            del llm
            return True

        except Exception as e:
            print(f"❌ {test['name']} - Failed: {e}")
            print(f"Error type: {type(e).__name__}")

    return False


def main():
    """Main diagnostic function."""
    print("🔍 Model Loading Diagnostic Tool")
    print("=" * 50)

    success = True

    # Check system
    if not check_system_info():
        success = False

    # Check model file
    if not check_model_file():
        success = False

    # Test model loading
    if success and not test_model_loading():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("✅ All checks passed! The model should work.")
    else:
        print("❌ Some checks failed. Review the errors above.")
        print("\nPossible solutions:")
        print("1. Try re-downloading the model (delete the file and restart)")
        print("2. Check if you have enough RAM (model needs ~4GB)")
        print("3. Update llama-cpp-python: pip install --upgrade llama-cpp-python")
        print("4. Try CPU-only mode by setting N_GPU_LAYERS=0 in your .env")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
