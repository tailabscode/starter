"""Structured JSON logging, one object per line, with no third-party dependency.

Level is controlled by the ``LOG_LEVEL`` environment variable (default ``INFO``). Never log
secrets: callers must not pass API keys or other credentials as log fields.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time


class JsonFormatter(logging.Formatter):
    """Renders each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "taskName",
                "message",
            )
        }
        payload.update(extras)
        return json.dumps(payload, default=str)


def configure_logging(name: str = "agentic_rag") -> logging.Logger:
    """Configure and return the package logger. Safe to call more than once."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    return logger
