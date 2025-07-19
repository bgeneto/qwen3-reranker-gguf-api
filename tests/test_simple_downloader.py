import pytest
import os
from unittest.mock import patch, MagicMock
from app.model_downloader import ensure_model_available, get_model_info
from app.config import settings


class TestSimpleDownloader:

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
                "model_filename", 
                "model_link",
                "exists_locally",
            ]

            for key in expected_keys:
                assert key in info

            assert info["local_path"] == settings.model_path
            assert info["model_filename"] == settings.MODEL_FILENAME
            assert info["model_link"] == settings.MODEL_LINK
            assert isinstance(info["exists_locally"], bool)

    def test_no_model_no_link_raises_error(self):
        """Test that missing model with no download link raises error"""
        with patch("os.path.exists", return_value=False), \
             patch.object(settings, 'MODEL_LINK', ''):
            with pytest.raises(RuntimeError, match="no MODEL_LINK provided"):
                ensure_model_available()

    @patch("subprocess.run")
    @patch("shutil.which")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("pathlib.Path.mkdir")
    def test_wget_download_success(self, mock_mkdir, mock_getsize, mock_exists, mock_which, mock_run):
        """Test successful download using wget"""
        # Setup mocks
        mock_exists.side_effect = [False, True]  # First call: file doesn't exist, second: it does
        mock_getsize.return_value = 1024  # Non-empty file
        mock_which.side_effect = lambda tool: "/usr/bin/wget" if tool == "wget" else None
        mock_run.return_value = MagicMock()
        
        with patch.object(settings, 'MODEL_LINK', 'https://example.com/model.gguf'):
            result = ensure_model_available()
            
        assert result == settings.model_path
        mock_run.assert_called_once()
        # Check that wget was called with correct arguments
        args = mock_run.call_args[0][0]
        assert args[0] == "wget"
        assert "-O" in args
        assert settings.model_path in args
        assert "https://example.com/model.gguf" in args

    @patch("subprocess.run")
    @patch("shutil.which")
    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("pathlib.Path.mkdir")
    def test_curl_download_fallback(self, mock_mkdir, mock_getsize, mock_exists, mock_which, mock_run):
        """Test fallback to curl when wget fails"""
        # Setup mocks
        mock_exists.side_effect = [False, True]  # First call: file doesn't exist, second: it does
        mock_getsize.return_value = 1024  # Non-empty file
        mock_which.side_effect = lambda tool: "/usr/bin/curl" if tool == "curl" else None
        mock_run.return_value = MagicMock()
        
        with patch.object(settings, 'MODEL_LINK', 'https://example.com/model.gguf'):
            result = ensure_model_available()
            
        assert result == settings.model_path
        mock_run.assert_called_once()
        # Check that curl was called with correct arguments
        args = mock_run.call_args[0][0]
        assert args[0] == "curl"
        assert "-L" in args
        assert "-o" in args
        assert settings.model_path in args
        assert "https://example.com/model.gguf" in args
