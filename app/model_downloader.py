import os
import logging
import subprocess
import shutil
import hashlib
from pathlib import Path
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)


def verify_gguf_file(file_path: str) -> bool:
    """
    Verify that a file is a valid GGUF file by checking its magic bytes and basic structure.

    Returns:
        bool: True if file appears to be a valid GGUF file
    """
    try:
        with open(file_path, "rb") as f:
            # Read first 4 bytes to check GGUF magic
            magic = f.read(4)
            # GGUF files start with 'GGUF' magic bytes
            if magic == b"GGUF":
                logger.info(f"File {file_path} has valid GGUF magic bytes")

                # Read version (4 bytes, little endian)
                version_bytes = f.read(4)
                if len(version_bytes) == 4:
                    version = int.from_bytes(version_bytes, byteorder="little")
                    logger.info(f"GGUF version: {version}")

                    # Basic version check - GGUF versions should be reasonable
                    if version > 0 and version < 10:
                        return True
                    else:
                        logger.warning(f"Unusual GGUF version: {version}")
                        return False
                else:
                    logger.error(f"Could not read GGUF version bytes")
                    return False
            else:
                logger.error(
                    f"File {file_path} does not have GGUF magic bytes. Found: {magic}"
                )
                return False
    except Exception as e:
        logger.error(f"Error verifying GGUF file {file_path}: {e}")
        return False


def calculate_file_hash(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return ""


def ensure_model_available() -> str:
    """
    Ensures the model is available locally. If not found and MODEL_LINK is provided,
    downloads using wget or curl.

    Returns:
        str: Path to the model file

    Raises:
        RuntimeError: If model cannot be downloaded or found
    """
    model_path = settings.model_path

    # Check if model already exists locally
    if os.path.exists(model_path):
        # Check if we can read the existing model file
        if not os.access(model_path, os.R_OK):
            raise RuntimeError(
                f"Model file exists at {model_path} but is not readable. "
                f"This is likely a permission issue. Please ensure the file has correct permissions. "
                f"You may need to set UID and GID in your .env file to match your host user."
            )

        # Verify the existing file is a valid GGUF file
        if not verify_gguf_file(model_path):
            logger.warning(
                f"Existing model file {model_path} appears to be corrupted or invalid, will re-download"
            )
            try:
                os.remove(model_path)
                logger.info(f"Removed corrupted model file: {model_path}")
            except Exception as e:
                logger.error(f"Failed to remove corrupted model file: {e}")
                raise RuntimeError(
                    f"Model file is corrupted and cannot be removed: {model_path}"
                )
        else:
            logger.info(f"Model found locally at: {model_path}")
            return model_path

    # If no MODEL_LINK provided, we can't download
    if not settings.MODEL_LINK:
        raise RuntimeError(
            f"Model not found at {model_path} and no MODEL_LINK provided for download"
        )

    logger.info(
        f"Model not found at {model_path}, attempting to download from: {settings.MODEL_LINK}"
    )

    # Create models directory if it doesn't exist
    model_dir = os.path.dirname(model_path)
    try:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        # Ensure we can write to the directory
        if not os.access(model_dir, os.W_OK):
            raise RuntimeError(
                f"Cannot write to models directory: {model_dir}. "
                f"This is likely a permission issue. Please ensure UID and GID in your .env file "
                f"match your host user, and that the models directory has correct permissions."
            )
    except PermissionError as e:
        raise RuntimeError(
            f"Permission denied creating models directory: {model_dir}. "
            f"Please ensure UID and GID in your .env file match your host user. Error: {e}"
        )

    # Try to download using wget first, then curl
    success = False

    # Try wget
    if shutil.which("wget"):
        try:
            logger.info("Downloading model using wget...")
            cmd = ["wget", "-O", model_path, settings.MODEL_LINK]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            success = True
            logger.info("Model successfully downloaded using wget")
        except subprocess.CalledProcessError as e:
            logger.warning(f"wget failed: {e.stderr}")

    # Try curl if wget failed or isn't available
    if not success and shutil.which("curl"):
        try:
            logger.info("Downloading model using curl...")
            cmd = ["curl", "-L", "-o", model_path, settings.MODEL_LINK]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            success = True
            logger.info("Model successfully downloaded using curl")
        except subprocess.CalledProcessError as e:
            logger.warning(f"curl failed: {e.stderr}")

    if not success:
        error_msg = (
            f"Failed to download model from {settings.MODEL_LINK}. "
            f"Please ensure wget or curl is installed and the URL is accessible."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Verify the downloaded file exists and is not empty
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Download completed but model file not found at: {model_path}"
        )

    file_size = os.path.getsize(model_path)
    if file_size == 0:
        os.remove(model_path)  # Remove empty file
        raise RuntimeError(f"Downloaded model file is empty")

    # Verify the downloaded file is a valid GGUF file
    if not verify_gguf_file(model_path):
        logger.error(f"Downloaded file {model_path} is not a valid GGUF file")
        # Calculate hash for debugging
        file_hash = calculate_file_hash(model_path)
        logger.error(f"File hash (SHA256): {file_hash}")
        try:
            os.remove(model_path)  # Remove invalid file
        except Exception as e:
            logger.error(f"Failed to remove invalid file: {e}")
        raise RuntimeError(f"Downloaded file is not a valid GGUF format")

    # Ensure the downloaded file has correct permissions (readable by owner and group)
    try:
        os.chmod(model_path, 0o644)
        logger.info(f"Set permissions on downloaded model file: {model_path}")
    except Exception as e:
        logger.warning(f"Could not set permissions on model file: {e}")

    # Calculate and log file hash for verification
    file_hash = calculate_file_hash(model_path)
    logger.info(
        f"Model successfully downloaded to: {model_path} (size: {file_size} bytes, SHA256: {file_hash[:16]}...)"
    )
    return model_path


def get_model_info() -> dict:
    """
    Returns information about the model configuration.

    Returns:
        dict: Model configuration information
    """
    return {
        "local_path": settings.model_path,
        "model_filename": settings.MODEL_FILENAME,
        "model_link": settings.MODEL_LINK,
        "exists_locally": os.path.exists(settings.model_path),
    }
