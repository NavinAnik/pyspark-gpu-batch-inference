from __future__ import annotations

import logging
from typing import Optional


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure a logger instance used across the project.

    Logging is intentionally centralized to ensure consistent formatting and
    to make log aggregation on clusters simpler (e.g., forwarding to CloudWatch,
    Stackdriver, or Azure Monitor).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Logger is already configured; avoid duplicate handlers when used
        # from multiple modules or in Spark worker processes.
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # In Spark workers, propagate is often disabled to prevent log duplication.
    logger.propagate = False

    return logger

