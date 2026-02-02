"""
Reasoning Extraction Module

This module provides functions for extracting reasoning text and final answers
from message content.
"""

import html
import re
from typing import Tuple

from open_webui.utils.middleware import DEFAULT_REASONING_TAGS


def strip_html(text: str) -> str:
    """
    Remove HTML tags from text.

    Args:
        text: Text containing HTML tags

    Returns:
        Cleaned text without HTML tags
    """
    cleaned = re.sub(r"<[^>]+>", "", text or "")
    return html.unescape(cleaned).strip()


def extract_reasoning_and_answer(content: str) -> Tuple[str, str]:
    """
    Extract reasoning text and final answer from message content.

    Args:
        content: Message content that may contain reasoning tags

    Returns:
        Tuple of (reasoning_text, final_answer)
    """
    reasoning_segments: list[str] = []

    # Extract from <details type="reasoning"> tags
    details_pattern = re.compile(
        r'<details\s+type="reasoning"[^>]*>(.*?)</details>',
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in details_pattern.finditer(content):
        inner = re.sub(
            r"<summary>.*?</summary>",
            "",
            match.group(1),
            flags=re.IGNORECASE | re.DOTALL,
        )
        cleaned = strip_html(inner)
        if cleaned:
            reasoning_segments.append(cleaned)

    # Extract from default reasoning tags
    for start_tag, end_tag in DEFAULT_REASONING_TAGS:
        pattern = re.compile(
            rf"{re.escape(start_tag)}(.*?){re.escape(end_tag)}", flags=re.DOTALL
        )
        for match in pattern.finditer(content):
            cleaned = strip_html(match.group(1))
            if cleaned:
                reasoning_segments.append(cleaned)

    # Deduplicate reasoning segments
    deduped = []
    seen = set()
    for segment in reasoning_segments:
        normalized = segment.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)

    reasoning_text = "\n\n".join(deduped)

    # Extract final answer (content without reasoning tags)
    without_details = details_pattern.sub("", content)
    for start_tag, end_tag in DEFAULT_REASONING_TAGS:
        without_details = re.sub(
            rf"{re.escape(start_tag)}(.*?){re.escape(end_tag)}",
            "",
            without_details,
            flags=re.DOTALL,
        )

    without_details = re.sub(
        r"<summary>.*?</summary>",
        "",
        without_details,
        flags=re.IGNORECASE | re.DOTALL,
    )
    final_answer = strip_html(without_details)

    return reasoning_text, final_answer


def extract_completion_content(response: dict) -> str:
    """
    Extract content from a chat completion response.

    Args:
        response: Chat completion response dictionary

    Returns:
        Extracted content string
    """
    try:
        return (
            response.get("choices", [])[0].get("message", {}).get("content", "") or ""
        )
    except Exception:
        return ""
