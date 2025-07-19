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
    Path(model_dir).mkdir(parents=True, exist_ok=True)

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
