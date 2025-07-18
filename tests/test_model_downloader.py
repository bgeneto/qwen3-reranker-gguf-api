import pytest
import os
from unittest.mock import patch, MagicMock
from app.model_downloader import ensure_model_available, get_model_info
from app.config import settings


class TestModelDownloader:

    def test_model_exists_locally(self):
        """Test that existing local model is returned without download"""
        with patch("os.path.exists", return_value=True):
            result = ensure_model_available()
            assert result == settings.model_path

    def test_model_info(self):
        """Test that model info returns correct structure"""
        with patch("os.path.exists", return_value=True):
            info = get_model_info()

            expected_keys = [
                "local_path",
                "hf_repo",
                "hf_filename",
                "exists_locally",
                "has_hf_token",
            ]

            for key in expected_keys:
                assert key in info

            assert info["local_path"] == settings.model_path
            assert info["hf_repo"] == settings.HF_MODEL_REPO
            assert info["hf_filename"] == settings.HF_MODEL_FILENAME
            assert isinstance(info["exists_locally"], bool)
            assert isinstance(info["has_hf_token"], bool)

    @patch("app.model_downloader.hf_hub_download")
    @patch("app.model_downloader.login")
    @patch("os.path.exists")
    @patch("pathlib.Path.mkdir")
    def test_model_download_success(
        self, mock_mkdir, mock_exists, mock_login, mock_download
    ):
        """Test successful model download"""
        # Setup mocks
        mock_exists.return_value = False
        mock_download.return_value = "/models/qwen3-4b-reranker-q4_k_m.gguf"

        # Mock settings for test
        with patch.object(settings, "HF_TOKEN", "test_token"):
            result = ensure_model_available()

        # Verify login was called with token
        mock_login.assert_called_once_with(
            token="test_token", add_to_git_credential=False
        )

        # Verify download was called with correct parameters
        mock_download.assert_called_once()

        # Verify result
        assert result == "/models/qwen3-4b-reranker-q4_k_m.gguf"

    @patch("app.model_downloader.hf_hub_download")
    @patch("os.path.exists")
    def test_model_download_no_token(self, mock_exists, mock_download):
        """Test model download without HF token"""
        mock_exists.return_value = False
        mock_download.return_value = "/models/qwen3-4b-reranker-q4_k_m.gguf"

        with patch.object(settings, "HF_TOKEN", ""):
            result = ensure_model_available()

        # Verify download was called with no token
        mock_download.assert_called_once()
        call_args = mock_download.call_args
        assert call_args.kwargs.get("token") is None

        assert result == "/models/qwen3-4b-reranker-q4_k_m.gguf"

    @patch("app.model_downloader.hf_hub_download")
    @patch("os.path.exists")
    def test_model_download_failure(self, mock_exists, mock_download):
        """Test model download failure handling"""
        from huggingface_hub.utils import HfHubHTTPError

        mock_exists.return_value = False
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_download.side_effect = HfHubHTTPError("Not found", response=mock_response)

        with pytest.raises(RuntimeError) as exc_info:
            ensure_model_available()

        assert "Model not found" in str(exc_info.value)
