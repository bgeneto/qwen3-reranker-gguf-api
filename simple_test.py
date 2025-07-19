#!/usr/bin/env python3
"""
Simple model loading test with different configurations.
This script will help identify the exact issue with model loading.
"""

import os
import sys

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))


def test_basic_load():
    """Test basic model loading with minimal configuration."""
    try:
        from llama_cpp import Llama
        from app.config import settings

        model_path = settings.model_path
        print(f"Attempting to load model from: {model_path}")

        # Test 1: Absolute minimal config
        print("\n--- Test 1: Minimal CPU-only configuration ---")
        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=0,
                n_ctx=512,
                n_batch=32,
                n_threads=1,
                logits_all=True,
                verbose=True,
                use_mmap=False,
                use_mlock=False,
            )
            print("✅ SUCCESS: Minimal configuration worked!")

            # Test tokenization
            tokens = llm.tokenize(b"test")
            print(f"✅ Tokenization test passed: {len(tokens)} tokens")

            del llm
            return True

        except Exception as e:
            print(f"❌ FAILED: {e}")
            print(f"Error type: {type(e).__name__}")

        # Test 2: Different memory settings
        print("\n--- Test 2: Different memory settings ---")
        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=0,
                n_ctx=1024,
                n_batch=64,
                n_threads=1,
                logits_all=True,
                verbose=True,
                use_mmap=True,
                use_mlock=True,
                f16_kv=False,
            )
            print("✅ SUCCESS: Memory settings configuration worked!")
            del llm
            return True

        except Exception as e:
            print(f"❌ FAILED: {e}")

        # Test 3: Check if it's a model format issue
        print("\n--- Test 3: Model format verification ---")
        with open(model_path, "rb") as f:
            magic = f.read(4)
            print(f"File magic bytes: {magic}")
            if magic == b"GGUF":
                version_bytes = f.read(4)
                version = int.from_bytes(version_bytes, byteorder="little")
                print(f"GGUF version: {version}")
            else:
                print("❌ Not a GGUF file!")
                return False

        return False

    except ImportError as e:
        print(f"❌ Cannot import required modules: {e}")
        return False


def main():
    print("🔧 Simple Model Loading Test")
    print("=" * 40)

    if test_basic_load():
        print("\n✅ Model loading successful!")
        print(
            "The issue might be with the specific configuration used in your main application."
        )
    else:
        print("\n❌ Model loading failed with all configurations.")
        print("\nPossible solutions:")
        print("1. The model file might be corrupted - try re-downloading")
        print("2. Insufficient memory - ensure you have at least 4GB free RAM")
        print("3. llama-cpp-python version incompatibility - try:")
        print("   pip install --upgrade llama-cpp-python")
        print("4. Try a different quantized version (e.g., Q4_0 instead of Q4_K_M)")


if __name__ == "__main__":
    main()
