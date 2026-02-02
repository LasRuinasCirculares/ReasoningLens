"""
Extractors Module

This module provides text extraction and processing utilities for reasoning analysis.
"""

from .reasoning import (
    extract_reasoning_and_answer,
    extract_completion_content,
    strip_html,
)
from .sections import (
    split_reasoning_into_sections,
    format_sections_for_prompt,
    format_sections_for_layer2_prompt,
    get_section_range_text,
    extract_segment_from_node_sections,
    is_continuation_section,
    StreamingReasoningSectionBuilder,
    CONTINUATION_WORDS,
)

__all__ = [
    # Reasoning extraction
    "extract_reasoning_and_answer",
    "extract_completion_content",
    "strip_html",
    # Section processing
    "split_reasoning_into_sections",
    "format_sections_for_prompt",
    "format_sections_for_layer2_prompt",
    "get_section_range_text",
    "extract_segment_from_node_sections",
    "is_continuation_section",
    "StreamingReasoningSectionBuilder",
    "CONTINUATION_WORDS",
]
