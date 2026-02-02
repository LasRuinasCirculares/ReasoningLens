"""
Layer 1 Analysis Module

This module handles coarse-grained (Layer 1) analysis of reasoning chains.
Layer 1 decomposes reasoning into high-level logical units: problem decomposition,
reasoning steps, intermediate answers, and final answers.
"""

import logging
from typing import Optional, List, Dict

from ..prompts import (
    detect_language,
    get_prompts,
)
from ..extractors.sections import (
    split_reasoning_into_sections,
    format_sections_for_prompt,
)
from ..extractors.reasoning import extract_completion_content
from ..utils.json_utils import safe_json_loads

log = logging.getLogger(__name__)


def build_layer1_messages(
    question: str,
    reasoning_text: str,
    final_answer: str,
    prompt_override: Optional[str] = None,
    sections: Optional[List[Dict]] = None,
    language: Optional[str] = None,
) -> list[dict]:
    """
    Build messages for Layer 1 (coarse-grained) analysis.

    Args:
        question: User's question
        reasoning_text: Extracted reasoning text
        final_answer: Final answer from the assistant
        prompt_override: Optional custom prompt template
        sections: Pre-split sections of the reasoning text
        language: Language code ('en' or 'zh'). Auto-detected if not provided.

    Returns:
        List of message dictionaries for the LLM
    """
    # Auto-detect language if not specified
    if language is None:
        language = detect_language(reasoning_text or question or "")
        log.info("Auto-detected language: %s", language)

    # Get language-specific prompts
    layer1_system, layer1_user, _, _ = get_prompts(language)

    prompt_template = prompt_override.strip() if prompt_override else layer1_user

    # Format sections for prompt
    reasoning_with_sections = format_sections_for_prompt(sections) if sections else ""
    section_count = len(sections) if sections else 0

    user_prompt = prompt_template.format(
        QUESTION=question or "Not available.",
        REASONING_WITH_SECTIONS=reasoning_with_sections or "Not available.",
        SECTION_COUNT=section_count,
        # Keep these for backward compatibility with custom prompts
        REASONING=reasoning_text or "Not available.",
        REASONING_LENGTH=len(reasoning_text or ""),
        FINAL_ANSWER=final_answer or "Not available.",  # For backward compatibility
    )

    return [
        {
            "role": "system",
            "content": layer1_system,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


async def analyze_layer1(
    question: str,
    reasoning_text: str,
    final_answer: str,
    run_stage_func,
    prompt_override: Optional[str] = None,
    sections: Optional[List[Dict]] = None,
    max_retries: int = 2,
) -> tuple:
    """
    Perform Layer 1 (coarse-grained) analysis with automatic retry on JSON parse failure.

    Args:
        question: User's question
        reasoning_text: Extracted reasoning text
        final_answer: Final answer from the assistant
        run_stage_func: Async function to run an analysis stage
        prompt_override: Optional custom prompt template
        sections: Pre-split sections of the reasoning text
        max_retries: Maximum number of retry attempts (default: 2)

    Returns:
        Tuple of (Layer 1 analysis result dict, raw response, sections list)
    """
    # Split reasoning into sections if not provided
    if sections is None:
        sections = split_reasoning_into_sections(reasoning_text)

    log.info("Split reasoning into %d sections", len(sections))

    messages = build_layer1_messages(
        question=question,
        reasoning_text=reasoning_text,
        final_answer=final_answer,
        prompt_override=prompt_override,
        sections=sections,
    )

    result = None
    response = None
    content = None

    # Retry loop with automatic fix attempts
    for attempt in range(max_retries + 1):
        if attempt > 0:
            log.warning("Layer 1 analysis retry attempt %d/%d", attempt, max_retries)

            # Add a correction prompt to the messages
            correction_prompt = (
                "\n\n⚠️ CRITICAL: Your previous response had invalid JSON syntax. "
                "Common errors to avoid:\n"
                "1. ALL string values MUST be wrapped in double quotes\n"
                '2. Example: "type": "reasoning" (NOT "type": reasoning)\n'
                "3. Ensure all commas are properly placed\n"
                "4. No trailing commas before closing braces\n\n"
                "Please regenerate the COMPLETE JSON response with valid syntax."
            )

            # Append correction to last user message
            messages[-1]["content"] += correction_prompt

        # Run the analysis stage
        response = await run_stage_func("layer1_coarse", messages)
        content = extract_completion_content(response)

        # Try to parse JSON (with automatic fixes on first attempt)
        result = safe_json_loads(content, auto_fix=(attempt == 0))

        if result is not None:
            if attempt > 0:
                log.info("Layer 1 JSON parsing succeeded on retry attempt %d", attempt)
            return result, response, sections

        # Log failure details
        log.error(
            "Layer 1 JSON parsing failed (attempt %d/%d). Raw content length: %d",
            attempt + 1,
            max_retries + 1,
            len(content) if content else 0,
        )
        if content and attempt == 0:  # Only log preview on first failure
            log.error("Raw content preview (first 500 chars): %s", content[:500])

    # All retries exhausted
    log.error("Layer 1 analysis failed after %d attempts", max_retries + 1)
    return result, response, sections
