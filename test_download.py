#!/usr/bin/env python3
"""
Simple test script to check if model download works
Run this outside of Docker to test the model download functionality
"""

import os
import sys
from pathlib import Path

# Set environment variables for testing
os.environ["HF_MODEL_REPO"] = "Mungert/Qwen3-Reranker-4B-GGUF"
os.environ["HF_MODEL_FILENAME"] = "Qwen3-Reranker-4B-q4_k_m.gguf"
os.environ["HF_TOKEN"] = ""  # Empty token to test without authentication
os.environ["LOG_LEVEL"] = "INFO"

# Add the app directory to Python path
sys.path.insert(0, "app")

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    from huggingface_hub.utils import HfHubHTTPError

    print("=== Testing HuggingFace Hub Access ===")

    repo = os.environ["HF_MODEL_REPO"]
    filename = os.environ["HF_MODEL_FILENAME"]

    print(f"Repository: {repo}")
    print(f"Filename: {filename}")
    print()

    # First, try to list files in the repository
    print("=== Listing repository files ===")
    try:
        files = list_repo_files(repo_id=repo, token=None)
        print(f"Found {len(files)} files in repository:")

        # Look for files that match our pattern
        matching_files = [
            f for f in files if filename.lower() in f.lower() or "q4_k_m" in f.lower()
        ]
        if matching_files:
            print("Files matching our search:")
            for f in matching_files:
                print(f"  - {f}")
        else:
            print("No files matching our search pattern found.")
            print("Available GGUF files:")
            gguf_files = [f for f in files if f.endswith(".gguf")]
            for f in gguf_files[:10]:  # Show first 10
                print(f"  - {f}")

    except Exception as e:
        print(f"Error listing files: {e}")
        print()

    # Try to download the specific file
    print("=== Testing download ===")
    try:
        # Create a test directory
        test_dir = Path("./test_models")
        test_dir.mkdir(exist_ok=True)

        print(f"Attempting to download {filename}...")
        downloaded_path = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(test_dir),
            token=None,
        )

        print(f"Success! Downloaded to: {downloaded_path}")

        # Check file size
        if os.path.exists(downloaded_path):
            size = os.path.getsize(downloaded_path)
            print(f"File size: {size:,} bytes ({size / (1024*1024*1024):.2f} GB)")

    except HfHubHTTPError as e:
        print(f"HTTP Error {e.response.status_code}: {e}")
        if e.response.status_code == 404:
            print("The file was not found. This could mean:")
            print("1. The filename is incorrect")
            print("2. The file doesn't exist in the repository")
            print("3. The repository name is wrong")
        elif e.response.status_code == 401:
            print("Authentication required. This model may need a HuggingFace token.")
    except Exception as e:
        print(f"Download error: {e}")

except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install huggingface_hub")
except Exception as e:
    print(f"Unexpected error: {e}")
