"""Consistent console and file logging for scripts and the dashboard."""

import logging

from config import PROJECT_ROOT


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring the root handlers once."""
    root = logging.getLogger()
    if not root.handlers:
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / "football_analysis.log", encoding="utf-8"),
            ],
        )
    return logging.getLogger(name)
