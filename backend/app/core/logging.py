"""
Centralized Logging Infrastructure.

Provides structured console output for debugging and auditing without print statements.
"""

import logging
import sys


def setup_logging() -> logging.Logger:
    """
    Configures and returns the system logger.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("finrisk_platform")
    logger.setLevel(logging.INFO)

    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)

    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger


logger = setup_logging()