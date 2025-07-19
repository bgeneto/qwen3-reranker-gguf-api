import os
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from .config import settings

logger = logging.getLogger(__name__)


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
        raise RuntimeError(f"Download completed but model file not found at: {model_path}")
    
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        os.remove(model_path)  # Remove empty file
        raise RuntimeError(f"Downloaded model file is empty")
    
    # Ensure the downloaded file has correct permissions (readable by owner and group)
    try:
        os.chmod(model_path, 0o644)
        logger.info(f"Set permissions on downloaded model file: {model_path}")
    except Exception as e:
        logger.warning(f"Could not set permissions on model file: {e}")
    
    logger.info(f"Model successfully downloaded to: {model_path} (size: {file_size} bytes)")
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
