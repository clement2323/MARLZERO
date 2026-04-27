"""Structured logging using loguru."""

import sys
import warnings
from typing import Literal

from loguru import logger as _logger

# Suppress Python warnings from third-party packages (gym, numpy, etc.) globally.
warnings.filterwarnings("ignore")

LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
_FORMAT_FILE = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"


def _own_code_filter(record: dict) -> bool:
    """Show only messages from our code and __main__; drop third-party noise."""
    name: str = record["name"]
    return name.startswith("morris_rl") or name in ("__main__", "__mp_main__")


def setup_logging(level: LogLevel = "INFO", log_file: str | None = None) -> None:
    """Configure loguru output to stderr and optionally a rotating file.

    Args:
        level: Minimum log level to emit.
        log_file: Optional path for file output. Rotated at 100 MB, kept 30 days.
    """
    _logger.remove()
    _logger.add(sys.stderr, level=level, format=_FORMAT, colorize=True, filter=_own_code_filter)
    if log_file is not None:
        _logger.add(
            log_file,
            level=level,
            format=_FORMAT_FILE,
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            filter=_own_code_filter,
        )


logger = _logger

__all__ = ["logger", "setup_logging", "LogLevel"]
