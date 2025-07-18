import os, json, logging, asyncio
from datetime import datetime
from typing import Any, Dict


class AsyncJSONFileHandler(logging.Handler):
    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename

    def emit(self, record: logging.LogRecord) -> None:
        payload: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "msg": record.getMessage(),
            "module": record.module,
            "lineno": record.lineno,
        }

        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # If we have a running loop, create the task
            asyncio.create_task(self._write(payload))
        except RuntimeError:
            # No running event loop, write synchronously
            self._write_sync(payload)

    async def _write(self, payload: Dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()

        def write_log():
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)

            # Write to file safely
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

        await loop.run_in_executor(None, write_log)

    def _write_sync(self, payload: Dict[str, Any]) -> None:
        """Synchronous version for when no event loop is running"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        # Write to file safely
        with open(self.filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


def setup_logging() -> None:
    from .config import settings

    logger = logging.getLogger()
    logger.setLevel(settings.LOG_LEVEL)
    # console
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)
    # file
    if settings.LOG_TO_FILE:
        logger.addHandler(AsyncJSONFileHandler(settings.LOG_FILE))
