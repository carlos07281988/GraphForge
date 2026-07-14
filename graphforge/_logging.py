# Copyright 2026 GraphForge Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logging configuration for GraphForge.

Provides ``configure_logging()`` as a single entry point to set up
structured, consistent logging across all framework modules.
"""

from __future__ import annotations

import logging
from typing import Optional


def configure_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt: str = "%H:%M:%S",
) -> None:
    """Configure GraphForge's default logging.

    Call once at application startup to enable structured log output
    from all framework modules.

    Parameters
    ----------
    level:
        Logging level (e.g. ``logging.DEBUG``, ``logging.INFO``).
    fmt:
        Log format string.
    datefmt:
        Date/time format string.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger = logging.getLogger("graphforge")
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)
    logger.info(
        "GraphForge logging configured at level %s",
        logging.getLevelName(level),
    )


def get_logger(name: str) -> logging.Logger:
    """Get a GraphForge child logger.

    All framework loggers live under the ``graphforge`` namespace so a
    single ``configure_logging()`` call controls all output.
    """
    return logging.getLogger(f"graphforge.{name}")


__all__ = ["configure_logging", "get_logger"]
