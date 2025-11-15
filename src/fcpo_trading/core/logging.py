from __future__ import annotations

import logging
from logging.config import dictConfig

from fcpo_trading.core.config import settings


def setup_logging() -> None:
    """Configure structured logging using dictConfig."""
    log_level = settings.log_level.upper()

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "format": (
                        '{"time": "%(asctime)s",'
                        ' "level": "%(levelname)s",'
                        ' "name": "%(name)s",'
                        ' "message": "%(message)s"}'
                    )
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                }
            },
            "root": {
                "level": log_level,
                "handlers": ["default"],
            },
        }
    )


logger = logging.getLogger("fcpo-trading")
