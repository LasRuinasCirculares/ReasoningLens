"""
Cache Utilities Module

This module provides caching and persistence utilities for analysis results.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .json_utils import stringify_for_prompt

log = logging.getLogger(__name__)


def get_message_analysis_dir(
    log_dir: Path,
    message_id: str,
    model: str,
) -> Path:
    """
    Get the directory for storing analysis results for a specific message and model.

    Args:
        log_dir: Base directory for analysis logs
        message_id: Message ID
        model: Model name used for analysis

    Returns:
        Path to the message-specific analysis directory
    """
    safe_message = re.sub(r"[^a-zA-Z0-9_-]+", "_", message_id)
    safe_model = re.sub(r"[^a-zA-Z0-9_-]+", "_", model)
    return log_dir / f"{safe_message}_{safe_model}"


def get_cached_analysis(
    log_dir: Path,
    message_id: str,
    model: str,
) -> Optional[dict]:
    """
    Try to load cached analysis result.

    Args:
        log_dir: Base directory for analysis logs
        message_id: Message ID
        model: Model name used for analysis

    Returns:
        Cached analysis result or None if not found
    """
    try:
        message_dir = get_message_analysis_dir(log_dir, message_id, model)
        result_file = message_dir / "final_merged_result.json"

        if result_file.exists():
            with result_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
                log.info(
                    "Loaded cached analysis from %s",
                    result_file,
                )
                return data
    except Exception:
        log.exception(
            "Failed to load cached analysis for message=%s model=%s",
            message_id,
            model,
        )
    return None


def persist_analysis_artifact(
    stage: str,
    payload,
    metadata: dict,
    log_dir: Path,
    enabled: bool = False,
):
    """
    Persist analysis outputs to JSON for debugging.

    Args:
        stage: Analysis stage name
        payload: Data to persist
        metadata: Metadata about the analysis
        log_dir: Directory to save logs
        enabled: Whether logging is enabled
    """
    if not enabled:
        return

    try:
        message_id = metadata.get("message_id", "unknown")
        model = metadata.get("model", "unknown")

        # Create message-specific directory
        message_dir = get_message_analysis_dir(log_dir, message_id, model)
        message_dir.mkdir(parents=True, exist_ok=True)

        # Save to message-specific directory
        file_path = message_dir / f"{stage}.json"

        serializable_payload = payload
        try:
            json.dumps(serializable_payload)
        except Exception:
            serializable_payload = stringify_for_prompt(payload)

        file_path.write_text(
            json.dumps(
                {
                    "metadata": metadata,
                    "stage": stage,
                    "payload": serializable_payload,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        log.info("Saved reasoning analysis artifact to %s", file_path)
    except Exception:
        log.exception(
            "Failed to persist reasoning analysis artifact for stage '%s'", stage
        )
