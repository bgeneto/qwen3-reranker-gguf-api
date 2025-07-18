#!/usr/bin/env python3
"""
Debug script to test configuration and model loading
"""
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, "/srv")

try:
    from app.config import settings
    from app.model_downloader import get_model_info, ensure_model_available
    from app.logger import setup_logging
    import logging

    setup_logging()
    logger = logging.getLogger(__name__)

    print("=== Configuration Debug ===")
    print(f"HF_MODEL_REPO: {settings.HF_MODEL_REPO}")
    print(f"HF_MODEL_FILENAME: {settings.HF_MODEL_FILENAME}")
    print(
        f"HF_TOKEN: {'***set***' if settings.HF_TOKEN and settings.HF_TOKEN != 'your-huggingface-token-here' else 'not set'}"
    )
    print(f"Model path: {settings.model_path}")
    print(f"N_CTX: {settings.N_CTX}")
    print(f"N_GPU_LAYERS: {settings.N_GPU_LAYERS}")
    print(f"N_BATCH: {settings.N_BATCH}")
    print(f"N_THREADS: {settings.N_THREADS}")
    print(f"LOG_FILE: {settings.LOG_FILE}")
    print()

    print("=== Model Info ===")
    model_info = get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")
    print()

    print("=== Model Directory Check ===")
    model_dir = Path(settings.model_path).parent
    print(f"Model directory: {model_dir}")
    print(f"Directory exists: {model_dir.exists()}")
    if model_dir.exists():
        print(f"Directory permissions: {oct(model_dir.stat().st_mode)[-3:]}")
        print(
            f"Directory contents: {list(model_dir.iterdir()) if model_dir.exists() else 'N/A'}"
        )
    print()

    print("=== Attempting Model Download ===")
    try:
        model_path = ensure_model_available()
        print(f"Model available at: {model_path}")

        # Check file details
        model_file = Path(model_path)
        if model_file.exists():
            print(f"File size: {model_file.stat().st_size} bytes")
            print(f"File permissions: {oct(model_file.stat().st_mode)[-3:]}")
        else:
            print("ERROR: Model file does not exist after download!")
    except Exception as e:
        print(f"ERROR: Failed to ensure model availability: {e}")
        import traceback

        traceback.print_exc()

except Exception as e:
    print(f"ERROR: Failed to import or run debug: {e}")
    import traceback

    traceback.print_exc()
