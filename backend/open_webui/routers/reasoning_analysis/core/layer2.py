"""
Layer 2 Analysis Module

This module handles fine-grained (Layer 2) analysis of reasoning chains.
Layer 2 refines individual Layer 1 nodes into more detailed sub-trees
when they contain distinct internal phases worth decomposing.
"""

import logging
from typing import Optional, List, Dict

from ..prompts import (
    detect_language,
    get_prompts,
)
from ..extractors.sections import (
    extract_segment_from_node_sections,
    format_sections_for_layer2_prompt,
)
from ..extractors.reasoning import extract_completion_content
from ..utils.json_utils import safe_json_loads

log = logging.getLogger(__name__)


def build_layer2_messages(
    question: str,
    segment_text: str,
    sections: Optional[List[Dict]] = None,
    section_start: int = 1,
    section_end: int = 1,
    parent_node_id: str = "",
    parent_node_type: str = "",
    language: Optional[str] = None,
) -> list[dict]:
    """
    Build messages for Layer 2 (fine-grained) analysis of a segment.

    Args:
        question: User's original question
        segment_text: Text of the segment to analyze
        sections: Full list of sections from Layer 1 analysis
        section_start: Starting section number (1-indexed, inclusive) from parent node
        section_end: Ending section number (1-indexed, inclusive) from parent node
        parent_node_id: ID of the parent Layer 1 node
        parent_node_type: Type of the parent Layer 1 node
        language: Language code ('en' or 'zh'). Auto-detected if not provided.

    Returns:
        List of message dictionaries for the LLM
    """
    # Auto-detect language if not specified
    if language is None:
        language = detect_language(segment_text or question or "")
        log.debug("Auto-detected language for Layer 2: %s", language)

    # Get language-specific prompts
    _, _, layer2_system, layer2_user = get_prompts(language)

    # Format sections with original section numbers
    if sections and section_start and section_end:
        segment_with_sections = format_sections_for_layer2_prompt(
            sections, section_start, section_end
        )
    else:
        # Fallback to segment text if sections not available
        segment_with_sections = segment_text or "Not available."

    # Calculate section_start + 1 for the template
    section_start_plus_1 = (
        section_start + 1 if section_start < section_end else section_start
    )

    user_prompt = layer2_user.format(
        QUESTION=question or "Not available.",
        SEGMENT_WITH_SECTIONS=segment_with_sections,
        SECTION_START=section_start,
        SECTION_START_PLUS_1=section_start_plus_1,
        SECTION_END=section_end,
        PARENT_NODE_ID=parent_node_id or "unknown",
        PARENT_NODE_TYPE=parent_node_type or "unknown",
    )

    return [
        {
            "role": "system",
            "content": layer2_system,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


async def analyze_layer2_for_node(
    node: dict,
    question: str,
    sections: Optional[List[Dict]],
    run_stage_func,
    reasoning_text: str = "",
    final_answer: str = "",
    max_retries: int = 2,
) -> Optional[dict]:
    """
    Perform Layer 2 (fine-grained) analysis for a single node with automatic retry.

    This function is designed to be called individually for each node,
    enabling progressive/streaming analysis results.

    Args:
        node: Layer 1 node to analyze
        question: User's original question
        sections: Pre-split sections of the reasoning text
        run_stage_func: Async function to run an analysis stage
        reasoning_text: Full reasoning text (for error detection context)
        final_answer: The final answer (for ineffective reflection detection)
        max_retries: Maximum number of retry attempts (default: 2)

    Returns:
        Dictionary with Layer 2 analysis results, or None if node cannot be analyzed
    """
    node_id = node.get("id", "")
    node_type = node.get("type", "")
    section_start = node.get("section_start", 0)
    section_end = node.get("section_end", 0)

    # Extract segment using section-based approach
    segment_text = ""
    start_pos = -1
    end_pos = -1

    if sections and section_start and section_end:
        segment_text, start_pos, end_pos = extract_segment_from_node_sections(
            sections, node
        )
        if segment_text:
            log.debug(
                "Extracted segment for node %s using sections %d-%d",
                node_id,
                section_start,
                section_end,
            )

    if not segment_text or start_pos == -1:
        log.warning(
            "Could not extract segment text for node %s (type: %s), skipping",
            node_id,
            node_type,
        )
        return None

    # Store positions and extracted text in the node for reference
    node["_segment_start"] = start_pos
    node["_segment_end"] = end_pos
    node["_segment_text"] = segment_text

    # Initialize result structure
    layer2_result = {
        "steps": [],
        "issues": [],
        "tree": {"nodes": [], "edges": []},
        "segment_text": segment_text,
        "segment_start": start_pos,
        "segment_end": end_pos,
        "section_start": section_start,
        "section_end": section_end,
        "can_be_refined": True,
        "refinement_reason": "",
        "responses": {},
    }

    # Run Layer 2 analysis with retry mechanism
    layer2_parsed = None
    layer2_response = None
    layer2_content = None

    try:
        layer2_msg = build_layer2_messages(
            question=question,
            segment_text=segment_text,
            sections=sections,
            section_start=section_start,
            section_end=section_end,
            parent_node_id=node_id,
            parent_node_type=node_type,
        )

        # Retry loop with automatic fix attempts
        for attempt in range(max_retries + 1):
            if attempt > 0:
                log.warning(
                    "Layer 2 analysis retry attempt %d/%d for node %s",
                    attempt,
                    max_retries,
                    node_id,
                )

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
                layer2_msg[-1]["content"] += correction_prompt

            layer2_response = await run_stage_func(f"layer2_{node_id}", layer2_msg)
            layer2_content = extract_completion_content(layer2_response)

            # Try to parse JSON (with automatic fixes on first attempt)
            layer2_parsed = safe_json_loads(layer2_content, auto_fix=(attempt == 0))

            if layer2_parsed is not None and isinstance(layer2_parsed, dict):
                if attempt > 0:
                    log.info(
                        "Layer 2 JSON parsing succeeded on retry attempt %d for node %s",
                        attempt,
                        node_id,
                    )
                break

            # Log failure details
            log.error(
                "Layer 2 JSON parsing failed for node %s (attempt %d/%d). Content length: %d",
                node_id,
                attempt + 1,
                max_retries + 1,
                len(layer2_content) if layer2_content else 0,
            )
            if layer2_content and attempt == 0:  # Only log preview on first failure
                log.error(
                    "Raw content preview (first 300 chars): %s", layer2_content[:300]
                )

        # Process the parsed result if successful
        if isinstance(layer2_parsed, dict):
            steps = layer2_parsed.get("steps", [])
            tree = (
                layer2_parsed.get("tree")
                if isinstance(layer2_parsed.get("tree"), dict)
                else {"nodes": [], "edges": []}
            )

            layer2_result["steps"] = steps
            layer2_result["tree"] = {
                "nodes": tree.get("nodes", []),
                "edges": tree.get("edges", []),
                "summary": tree.get("summary", ""),
            }
            layer2_result["can_be_refined"] = layer2_parsed.get(
                "can_be_refined", bool(steps) or bool(layer2_result["tree"]["nodes"])
            )
            layer2_result["refinement_reason"] = layer2_parsed.get(
                "refinement_reason", ""
            )
            layer2_result["responses"]["layer2"] = layer2_response
        else:
            log.error(
                "Layer 2 analysis failed for node %s after %d attempts",
                node_id,
                max_retries + 1,
            )

    except Exception as exc:
        log.warning("Layer 2 analysis exception for node %s: %s", node_id, exc)

    # Note: Error detection is now handled at a higher level by SectionAnalysisAgent
    # which processes all sections in batches with LLM-based detection

    return layer2_result
