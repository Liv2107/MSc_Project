"""Run-scoped logging and portable run identifiers."""

from __future__ import annotations

import logging
import re
import secrets
from datetime import UTC, datetime
from pathlib import Path


def configure_logging(*, run_id: str, log_path: Path, verbose: bool = False) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"ai_detector.{run_id}")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    # The FILE keeps the full, unambiguous audit record: absolute timestamp, run id, and
    # level on every line.
    file_formatter = logging.Formatter(
        fmt=f"%(asctime)s | {run_id} | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # The CONSOLE is for a human watching a long run, so it drops what is constant for the
    # whole run. The run id appeared twice per line (literally and as the logger name),
    # which pushed the actual message off the right of the terminal.
    console_formatter = logging.Formatter(fmt="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(file_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("run %s | log file %s", run_id, log_path)
    return logger


def make_run_id(*, experiment_type: str, config_hash: str) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", experiment_type.strip()).strip("-_").lower()
    if not safe_name:
        raise ValueError("experiment_type does not contain a portable name")
    safe_hash = re.sub(r"[^a-fA-F0-9]", "", config_hash)[:10].lower()
    if len(safe_hash) < 6:
        raise ValueError("config_hash must contain at least six hexadecimal characters")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{safe_name}-{timestamp}-{safe_hash}-{secrets.token_hex(2)}"
