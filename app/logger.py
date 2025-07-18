import os, json, logging, asyncio
from datetime import datetime
from typing import Any, Dict

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
LOG_FILE = os.getenv("LOG_FILE", "/var/log/reranker.jsonl")


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
        asyncio.create_task(self._write(payload))

    async def _write(self, payload: Dict[str, Any]) -> None:
        loop = asyncio.get_running_loop()

        def write_log():
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)

            # Write to file safely
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload) + "\n")

        await loop.run_in_executor(None, write_log)


def setup_logging() -> None:
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVEL)
    # console
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console)
    # file
    if LOG_TO_FILE:
        logger.addHandler(AsyncJSONFileHandler(LOG_FILE))
