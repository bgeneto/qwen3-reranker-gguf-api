import os
import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download, login
from huggingface_hub.utils import HfHubHTTPError

from .config import settings

logger = logging.getLogger(__name__)


def ensure_model_available() -> str:
    """
    Ensures the model is available locally. If not found, downloads from Hugging Face.

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

    logger.info(
        f"Model not found at {model_path}, attempting to download from Hugging Face..."
    )

    # Login to Hugging Face if token is provided
    if settings.HF_TOKEN:
        try:
            login(token=settings.HF_TOKEN, add_to_git_credential=False)
            logger.info("Successfully authenticated with Hugging Face")
        except Exception as e:
            logger.warning(f"Failed to authenticate with Hugging Face: {e}")

    try:
        # Create models directory if it doesn't exist
        model_dir = os.path.dirname(model_path)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Download the model
        logger.info(
            f"Downloading {settings.HF_MODEL_FILENAME} from {settings.HF_MODEL_REPO}..."
        )

        downloaded_path = hf_hub_download(
            repo_id=settings.HF_MODEL_REPO,
            filename=settings.HF_MODEL_FILENAME,
            local_dir=model_dir,
            local_dir_use_symlinks=False,  # Copy file instead of symlinking
            token=settings.HF_TOKEN if settings.HF_TOKEN else None,
        )

        # If the downloaded file is not at the expected path, move it
        expected_path = os.path.join(model_dir, settings.HF_MODEL_FILENAME)
        if downloaded_path != expected_path and downloaded_path != model_path:
            import shutil

            shutil.move(downloaded_path, model_path)
            logger.info(f"Moved downloaded model to: {model_path}")
        else:
            model_path = downloaded_path

        logger.info(f"Model successfully downloaded to: {model_path}")
        return model_path

    except HfHubHTTPError as e:
        if e.response.status_code == 401:
            error_msg = (
                f"Authentication failed. Please check your HF_TOKEN. " f"Error: {e}"
            )
        elif e.response.status_code == 404:
            error_msg = (
                f"Model not found. Please check HF_MODEL_REPO ({settings.HF_MODEL_REPO}) "
                f"and HF_MODEL_FILENAME ({settings.HF_MODEL_FILENAME}). Error: {e}"
            )
        else:
            error_msg = f"HTTP error downloading model: {e}"

        logger.error(error_msg)
        raise RuntimeError(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error downloading model: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def get_model_info() -> dict:
    """
    Returns information about the model configuration.

    Returns:
        dict: Model configuration information
    """
    return {
        "local_path": settings.model_path,
        "hf_repo": settings.HF_MODEL_REPO,
        "hf_filename": settings.HF_MODEL_FILENAME,
        "exists_locally": os.path.exists(settings.model_path),
        "has_hf_token": bool(settings.HF_TOKEN),
    }
