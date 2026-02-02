"""
JSON Utilities Module

This module provides JSON parsing and serialization utilities.
"""

import json
import logging
import re
from typing import Optional

log = logging.getLogger(__name__)


def _try_fix_json_syntax(text: str) -> Optional[str]:
    """
    Attempt to automatically fix common JSON syntax errors.

    Common fixes:
    1. Missing quotes around string values: "type":value → "type":"value"
    2. Missing commas between properties
    3. Trailing commas before closing braces/brackets

    Args:
        text: Potentially malformed JSON text

    Returns:
        Fixed JSON text or None if fix is too risky
    """
    original = text

    # Fix 1: Missing quotes around common string values after colons
    # Pattern: "key":value (where value is alphanumeric/underscore and not a number/boolean/null)
    # This matches cases like "type":reasoning and fixes to "type":"reasoning"
    text = re.sub(
        r":\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}\]])",
        lambda m: (
            f': "{m.group(1)}"{m.group(2)}'
            if m.group(1) not in ["true", "false", "null"]
            else m.group(0)
        ),
        text,
    )

    # Fix 2: Remove trailing commas before closing braces/brackets
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    if text != original:
        log.info("Applied automatic JSON syntax fixes")
        return text

    return None


def safe_json_loads(text: str, auto_fix: bool = True) -> Optional[dict]:
    """
    Safely parse JSON text, handling various input types.

    Supports:
    - Plain JSON strings
    - JSON wrapped in markdown code blocks (```json ... ```)
    - Dict objects (passed through)
    - Automatic fixing of common JSON syntax errors

    Args:
        text: JSON text to parse
        auto_fix: Whether to attempt automatic fixes on parse failure

    Returns:
        Parsed dictionary or None if parsing fails
    """
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None

    # Strip whitespace
    text = text.strip()

    # Handle markdown code blocks: ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        # Remove opening marker (```json or ```)
        lines = text.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]  # Remove first line

        # Remove closing marker (```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # Remove last line

        text = "\n".join(lines).strip()

    # Try normal parsing first
    try:
        return json.loads(text)
    except Exception as e:
        log.debug("Initial JSON parsing failed: %s", e)

        # Try automatic fixes if enabled
        if auto_fix:
            fixed_text = _try_fix_json_syntax(text)
            if fixed_text:
                try:
                    result = json.loads(fixed_text)
                    log.info("Successfully parsed JSON after automatic fixes")
                    return result
                except Exception as fix_e:
                    log.debug("JSON parsing still failed after fixes: %s", fix_e)

        log.debug("JSON parsing failed: %s | Content preview: %s", e, text[:200])
        return None


def stringify_for_prompt(data: Optional[dict | str]) -> str:
    """
    Convert data to string format suitable for prompts.

    Args:
        data: Data to stringify

    Returns:
        String representation of data
    """
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return str(data)
