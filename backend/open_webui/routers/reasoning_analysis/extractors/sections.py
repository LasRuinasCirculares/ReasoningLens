"""
Section Processing Module

This module provides functions for splitting reasoning text into sections
and managing section-based text processing.
"""

import re
from typing import List, Dict, Tuple


# ============================================================================
# Continuation Words Configuration
# ============================================================================

# Words/phrases that indicate a section is a continuation of the previous one.
# If a section's first sentence starts with any of these, it should be merged
# with the previous section.
CONTINUATION_WORDS = [
    # Reconsideration / backtracking
    "Another",
    "Backtrack",
    "Going back",
    "Trace back",
    # Verification / checking
    "Check",
    "Let me check",
    "Let me verify",
    "Just to be thorough",
    "Let me just double-check",
    "Just to make sure",
    "Recheck",
    # Hesitation / uncertainty
    "Hmmm",
    "Hmm",
    "Maybe",
    "Might",
    "Perhaps",
    "Not sure",
    # Alternatives
    "Instead of",
    "Maybe I should consider",
    "Let me try another",
    "Maybe I can consider",
    # Contradiction / correction
    "However",
    "But",
    "Wait",
    "Hold on",
    # Retry
    "Retry",
]

# Compile regex pattern for efficient matching (case-insensitive)
_CONTINUATION_PATTERN = re.compile(
    r"^\s*(" + "|".join(re.escape(word) for word in CONTINUATION_WORDS) + r")\b",
    re.IGNORECASE,
)


def is_continuation_section(text: str) -> bool:
    """
    Check if a section starts with a continuation word/phrase.

    Args:
        text: The section text to check

    Returns:
        True if the section starts with a continuation word
    """
    if not text:
        return False

    # Get the first sentence (up to first period, question mark, or exclamation)
    first_sentence_match = re.match(r"^[^.!?\n]*[.!?]?", text.strip())
    if first_sentence_match:
        first_sentence = first_sentence_match.group(0)
    else:
        first_sentence = text.strip()[:100]  # Fallback to first 100 chars

    return bool(_CONTINUATION_PATTERN.match(first_sentence))


# ============================================================================
# Streaming Section Builder
# ============================================================================


class StreamingReasoningSectionBuilder:
    """
    A class for building reasoning sections incrementally during streaming output.

    This class processes reasoning content as it arrives during streaming,
    splitting it into sections based on \\n\\n delimiters. When finalized,
    it applies continuation word merging only if the result has more than 10 sections.

    Usage:
        builder = StreamingReasoningSectionBuilder()

        # During streaming, append content as it arrives
        builder.append("First part of reasoning...")
        builder.append("...more content\\n\\nNext section...")

        # When streaming is done, finalize and get sections
        sections = builder.finalize()
    """

    def __init__(self, min_section_threshold: int = 10):
        """
        Initialize the streaming section builder.

        Args:
            min_section_threshold: Minimum number of sections to apply continuation word merging.
                                   If section count > threshold, merge by continuation words.
                                   If section count <= threshold, keep raw \\n\\n splits.
        """
        self.min_section_threshold = min_section_threshold
        self.sections: List[Dict] = []
        self.pending_content: str = ""
        self.total_content: str = ""
        self._section_counter: int = 0

    def _create_section(self, text: str) -> Dict:
        """
        Create a new section dictionary.

        Args:
            text: The section text content

        Returns:
            Section dictionary with section_id, text, start_pos, end_pos
        """
        self._section_counter += 1
        start_pos = len(self.total_content) - len(text) - len(self.pending_content)
        if start_pos < 0:
            start_pos = 0

        return {
            "section_id": self._section_counter,
            "text": text.strip(),
            "start_pos": start_pos,
            "end_pos": start_pos + len(text),
        }

    def _add_new_section(self, text: str) -> None:
        """
        Add a new section to the sections list.

        Args:
            text: The section text content
        """
        section = self._create_section(text)
        if section["text"]:
            self.sections.append(section)

    def _process_pending_section(self, section_text: str) -> None:
        """
        Process a complete section that ends with \\n\\n.

        Simply adds each section without merging during streaming.
        Merging by continuation words happens in finalize() if needed.

        Args:
            section_text: The complete section text
        """
        section_text = section_text.strip()
        if not section_text:
            return

        self._add_new_section(section_text)

    def append(self, content: str) -> None:
        """
        Append new content to the builder.

        This method processes the content incrementally, splitting on \\n\\n
        and building sections as content arrives.

        Args:
            content: New content chunk to append
        """
        if not content:
            return

        # Add to total content for position tracking
        self.total_content += content

        # Add to pending content for processing
        self.pending_content += content

        # Process any complete sections (ended by \n\n)
        while "\n\n" in self.pending_content:
            # Find the first \n\n
            split_pos = self.pending_content.find("\n\n")

            # Extract the complete section (before \n\n)
            complete_section = self.pending_content[:split_pos]

            # Process this section
            self._process_pending_section(complete_section)

            # Keep the rest as pending (after \n\n)
            self.pending_content = self.pending_content[split_pos + 2 :]

    def _merge_by_continuation_words(self, sections: List[Dict]) -> List[Dict]:
        """
        Merge sections that start with continuation words with the previous section.

        Delegates to the module-level function to avoid code duplication.

        Args:
            sections: List of raw sections

        Returns:
            List of merged sections
        """
        return _merge_sections_by_continuation_words(sections)

    def finalize(self) -> List[Dict]:
        """
        Finalize the section building and return all sections.

        This should be called when streaming is complete to process
        any remaining pending content.

        Logic:
        1. First collect all sections split by \\n\\n
        2. Try merging by continuation words (wait, but, however, etc.)
        3. If merged count > threshold, use merged result
        4. If merged count <= threshold, use raw \\n\\n splits

        Returns:
            List of section dictionaries with keys:
            - section_id: int (1-indexed)
            - text: str (the section content)
            - start_pos: int (character position in original text)
            - end_pos: int (character position in original text)
        """
        # Process any remaining pending content
        if self.pending_content.strip():
            self._process_pending_section(self.pending_content)
            self.pending_content = ""

        # Try merging by continuation words
        merged_sections = self._merge_by_continuation_words(self.sections)

        # Decide which result to use based on count threshold
        if len(merged_sections) > self.min_section_threshold:
            # Use merged result (more than 10 sections)
            result_sections = merged_sections
        else:
            # Use raw \n\n splits (10 or fewer sections after merging)
            result_sections = self.sections

        # Re-number sections to ensure consecutive IDs
        for idx, section in enumerate(result_sections, start=1):
            section["section_id"] = idx

        return result_sections

    def get_current_sections(self) -> List[Dict]:
        """
        Get the current sections without finalizing.

        This is useful for getting intermediate results during streaming.

        Returns:
            List of currently completed sections
        """
        return self.sections.copy()

    def reset(self) -> None:
        """
        Reset the builder to its initial state.
        """
        self.sections = []
        self.pending_content = ""
        self.total_content = ""
        self._section_counter = 0


# ============================================================================
# Section-based Text Processing
# ============================================================================


def _detect_paragraph_delimiter(text: str) -> str:
    """
    Detect the best paragraph delimiter for the given text.

    Checks for various paragraph separation patterns:
    1. Standard double newlines: \\n\\n
    2. Markdown blockquote paragraph breaks: \\n> \\n> or \\n>\\n>
    3. Fall back to single newline if no paragraph markers found

    Returns:
        The detected delimiter pattern string
    """
    # Count occurrences of different paragraph patterns

    # Pattern 1: Standard double newlines (not inside blockquotes)
    # We need to check if the text uses blockquote format
    lines = text.split("\n")
    is_blockquote_format = (
        sum(1 for line in lines if line.strip().startswith(">")) > len(lines) * 0.5
    )

    if is_blockquote_format:
        # For blockquote format, paragraph breaks are empty quote lines
        # Pattern: line ends, then "> " alone or ">" alone, then next paragraph
        # We'll use a special marker to indicate blockquote paragraph mode
        return "BLOCKQUOTE_PARAGRAPH"

    # Pattern 2: Standard double newlines
    double_newline_count = text.count("\n\n")

    # If we have meaningful double newlines (more than just 1-2 at the edges)
    if double_newline_count >= 3:
        return "\n\n"

    # Fall back to single newline
    return "\n"


def _split_blockquote_paragraphs(text: str) -> List[str]:
    """
    Split blockquote-formatted text into paragraphs.

    Blockquote paragraphs are separated by empty quote lines (just "> " or ">").

    Args:
        text: Text in blockquote format (lines starting with >)

    Returns:
        List of paragraph strings (with > prefixes removed)
    """
    lines = text.split("\n")
    paragraphs = []
    current_paragraph_lines = []

    for line in lines:
        # Remove the > prefix if present
        stripped = line.strip()
        if stripped.startswith(">"):
            content = stripped[1:].strip()  # Remove > and leading space
        else:
            content = stripped

        # Check if this is an empty line (paragraph separator)
        if not content:
            # End current paragraph if we have content
            if current_paragraph_lines:
                paragraphs.append("\n".join(current_paragraph_lines))
                current_paragraph_lines = []
        else:
            current_paragraph_lines.append(content)

    # Don't forget the last paragraph
    if current_paragraph_lines:
        paragraphs.append("\n".join(current_paragraph_lines))

    return paragraphs


# ============================================================================
# Shared Section Merging Logic
# ============================================================================


def _merge_sections_by_continuation_words(sections: List[Dict]) -> List[Dict]:
    """
    Merge sections that start with continuation words with the previous section.

    This shared function is used by both StreamingReasoningSectionBuilder.finalize()
    and split_reasoning_into_sections() to avoid code duplication.

    Args:
        sections: List of raw section dictionaries with 'text', 'start_pos', 'end_pos' keys

    Returns:
        List of merged sections where continuation sections are combined with previous
    """
    if not sections:
        return []

    merged = []
    for section in sections:
        if not merged:
            # First section, just add it
            merged.append(section.copy())
        elif is_continuation_section(section["text"]):
            # This section starts with a continuation word, merge with previous
            prev_section = merged[-1]
            # Merge with \n\n separator
            prev_section["text"] = (
                prev_section["text"].strip() + "\n\n" + section["text"].strip()
            )
            prev_section["end_pos"] = section["end_pos"]
        else:
            # Regular section, add as new
            merged.append(section.copy())

    return merged


def split_reasoning_into_sections(
    reasoning_text: str, min_section_threshold: int = 10
) -> List[Dict]:
    """
    Split reasoning text into sections with conditional continuation word merging.

    Splitting strategy:
    1. Detect if text is in blockquote format (lines starting with >)
       - If so, split by empty quote lines (paragraph breaks in blockquotes)
    2. If text contains meaningful '\\n\\n' (double newlines), split by '\\n\\n'
    3. Otherwise, fall back to splitting by single '\\n'
    4. Try merging sections that start with continuation words (Wait, But, However, etc.)
    5. If merged section count > threshold, use merged result
    6. If merged section count <= threshold, use raw splits (no merging)

    Each section is numbered and contains its text and position info.

    Args:
        reasoning_text: Full reasoning text
        min_section_threshold: Minimum number of sections to apply continuation word merging.
                               If merged count > threshold, use merged result.
                               If merged count <= threshold, use raw splits.
                               Default is 10.

    Returns:
        List of section dictionaries with keys:
        - section_id: int (1-indexed)
        - text: str (the section content)
        - start_pos: int (character position in original text)
        - end_pos: int (character position in original text)
    """
    if not reasoning_text:
        return []

    # Detect the best delimiter
    delimiter = _detect_paragraph_delimiter(reasoning_text)

    if delimiter == "BLOCKQUOTE_PARAGRAPH":
        # Special handling for blockquote format
        parts = _split_blockquote_paragraphs(reasoning_text)
        delimiter_len = 1  # Approximate for position tracking
    elif delimiter == "\n\n":
        parts = reasoning_text.split("\n\n")
        delimiter_len = 2
    else:
        parts = reasoning_text.split("\n")
        delimiter_len = 1

    current_pos = 0

    # First pass: create raw sections from \n\n splits
    raw_sections = []
    for i, part in enumerate(parts):
        part_stripped = part.strip()
        if not part_stripped:
            # Skip empty sections but update position
            if i < len(parts) - 1:
                current_pos += len(part) + delimiter_len
            else:
                current_pos += len(part)
            continue

        # For blockquote format, we need to find the text differently
        # since we've removed the > prefixes
        if delimiter == "BLOCKQUOTE_PARAGRAPH":
            # Try to find the first few words in the original text
            search_text = part_stripped[: min(50, len(part_stripped))]
            start_pos = reasoning_text.find(search_text, current_pos)
            if start_pos == -1:
                # Try finding without > prefix consideration
                start_pos = current_pos
        else:
            # Find the actual position of this part in the original text
            start_pos = reasoning_text.find(part, current_pos)
            if start_pos == -1:
                start_pos = current_pos

        end_pos = start_pos + len(part)

        raw_sections.append(
            {
                "text": part_stripped,
                "start_pos": start_pos,
                "end_pos": end_pos,
            }
        )

        # Update position
        if i < len(parts) - 1:
            current_pos = end_pos + delimiter_len
        else:
            current_pos = end_pos

    # Second pass: merge sections that start with continuation words using shared function
    continuation_merged = _merge_sections_by_continuation_words(raw_sections)

    # Third pass: decide which result to use based on count threshold
    if len(continuation_merged) > min_section_threshold:
        # Use merged result (more than threshold sections)
        result_sections = continuation_merged
    else:
        # Use raw \n\n splits (threshold or fewer sections after merging)
        result_sections = raw_sections

    # Fourth pass: assign section IDs
    for section_id, section in enumerate(result_sections, start=1):
        section["section_id"] = section_id

    return result_sections


def format_sections_for_prompt(sections: List[Dict]) -> str:
    """
    Format sections into a labeled string for the LLM prompt.

    Args:
        sections: List of section dictionaries from split_reasoning_into_sections

    Returns:
        Formatted string with section labels
    """
    if not sections:
        return ""

    formatted_parts = []
    for section in sections:
        section_id = section["section_id"]
        text = section["text"]
        formatted_parts.append(f"[Section {section_id}]\n{text}")

    return "\n\n".join(formatted_parts)


def format_sections_for_layer2_prompt(
    sections: List[Dict], section_start: int, section_end: int
) -> str:
    """
    Format sections for Layer 2 prompt, preserving original section numbers.

    This ensures Layer 2 uses the same section numbering as Layer 1,
    preventing confusion when the model outputs section references.

    Args:
        sections: Full list of section dictionaries from Layer 1
        section_start: Starting section number (1-indexed, inclusive)
        section_end: Ending section number (1-indexed, inclusive)

    Returns:
        Formatted string with original section labels [Section N]
    """
    if not sections:
        return ""

    # Validate range
    section_start = max(1, section_start)
    section_end = min(len(sections), section_end)

    if section_start > section_end:
        return ""

    formatted_parts = []
    for i in range(section_start - 1, section_end):
        section = sections[i]
        section_id = section.get("section_id", i + 1)
        text = section.get("text", "")
        formatted_parts.append(f"[Section {section_id}]\n{text}")

    return "\n\n".join(formatted_parts)


def get_section_range_text(
    sections: List[Dict], section_start: int, section_end: int
) -> Tuple[str, int, int]:
    """
    Extract the combined text and positions for a range of sections.

    Args:
        sections: List of section dictionaries
        section_start: Starting section number (1-indexed)
        section_end: Ending section number (1-indexed, inclusive)

    Returns:
        Tuple of (combined_text, start_position, end_position)
    """
    if not sections:
        return "", -1, -1

    # Validate range
    section_start = max(1, section_start)
    section_end = min(len(sections), section_end)

    if section_start > section_end:
        return "", -1, -1

    # Get sections in range (convert to 0-indexed)
    selected_sections = sections[section_start - 1 : section_end]

    if not selected_sections:
        return "", -1, -1

    # Combine text
    combined_text = "\n\n".join(s["text"] for s in selected_sections)
    start_pos = selected_sections[0]["start_pos"]
    end_pos = selected_sections[-1]["end_pos"]

    return combined_text, start_pos, end_pos


def extract_segment_from_node_sections(
    sections: List[Dict], node: dict
) -> Tuple[str, int, int]:
    """
    Extract segment text from a node using section_start and section_end.

    Args:
        sections: List of section dictionaries
        node: Node dictionary with section_start and section_end fields

    Returns:
        Tuple of (extracted_text, start_position, end_position)
    """
    section_start = node.get("section_start", 0)
    section_end = node.get("section_end", 0)

    if not section_start or not section_end:
        # Fallback to old marker-based extraction
        return "", -1, -1

    return get_section_range_text(sections, section_start, section_end)
