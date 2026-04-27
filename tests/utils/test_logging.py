"""Smoke tests for logging setup."""

from morris_rl.utils.logging import logger, setup_logging


def test_setup_logging_does_not_crash() -> None:
    """setup_logging() completes without raising for all common levels."""
    setup_logging(level="DEBUG")
    setup_logging(level="INFO")
    setup_logging(level="WARNING")


def test_logger_is_usable() -> None:
    """The re-exported logger object is callable without error."""
    logger.info("smoke test log message")
