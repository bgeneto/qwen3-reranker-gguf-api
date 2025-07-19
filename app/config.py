from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Model
    MODEL_FILENAME: str = "Qwen3-Reranker-4B-q4_k_m.gguf"  # Default filename
    MODEL_LINK: str = ""  # Direct download URL for the model
    N_CTX: int = 8192
    N_GPU_LAYERS: int = -1
    N_BATCH: int = 512
    N_THREADS: int = 0

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_TO_FILE: bool = True
    LOG_FILE: str = "/var/log/reranker.jsonl"

    # Auth
    API_TOKEN: str = "change-me-please"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    @property
    def model_path(self) -> str:
        """Construct the full model path from the filename"""
        # Use /models directory in container, ./models locally
        return f"/models/{self.MODEL_FILENAME}"


settings = Settings()
