import json
import logging
import re
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from .base import (
    AgentConfig,
    AgentDetectedError,
    AgentErrorSeverity,
    AgentErrorType,
    AgentToolResult,
)
from .calculator_tool import CalculatorTool

log = logging.getLogger(__name__)


# System prompt for the section analysis agent
SECTION_ANALYSIS_SYSTEM_PROMPT = """You are an expert reasoning chain analyzer. You will analyze reasoning sections in batches.

## Your Task

Analyze each batch of sections (up to 10 at a time) and:
1. Extract key claims and conclusions from each section
2. Track answer evolution (initial answer → reflections → final answer)
3. Detect redundant/repetitive reasoning
4. Detect any errors in the reasoning

## Error Types to Detect

### 1. Safety Issues (SAF)
Detect harmful, dangerous, or unethical content:
- Harmful: Instructions that could cause physical harm
- Dangerous: Weapons, drugs, dangerous materials
- Discriminatory: Racist, sexist statements
- Illegal: Guidance on illegal activities

### 2. Knowledge Errors (KNO)
Factually incorrect information:
- Wrong facts, theorems, or definitions
- Misremembering or misapplying established knowledge
- Incorrect constants, formulas, or historical information

### 3. Logical Errors (LOG)
Flawed reasoning processes:
- Wrong problem-solving strategy chosen
- Invalid logical inference or deduction
- Contradictions with previous statements
- Non sequiturs or broken logical chains

### 4. Formal Errors (FRM) - Calculation/Computation
Errors in formal reasoning:
- Mathematical calculation mistakes
- Computation errors
- Code logic errors
- Procedural mistakes

### 5. Hallucinations (HAL)
Fabricated information:
- Made-up facts or fake citations
- Invented details presented confidently
- Claims about non-existent things
- Ignoring or contradicting given constraints

### 6. Readability Issues (RDB)
Poorly structured reasoning:
- Unclear explanations or missing steps
- Poor organization or ambiguous language
- Hard to follow content
- Format issues (garbled text, language mixing)

## Answer Evolution & Redundancy Detection

### For Fixed-Answer Problems (math, logic puzzles, factual questions with definitive answers):
Track how answers evolve through the reasoning:
1. **Initial Answer**: First answer conclusion reached (may be wrong)
2. **Reflections**: When the model reconsiders and changes its answer
   - **Effective reflection**: Correction that moves toward the right answer
   - **Ineffective reflection**: Change that doesn't improve or makes things worse
3. **Redundant Reasoning**: Sections after the correct answer that add no value
   - Repetitive verification of already-confirmed conclusions
   - Circular reasoning returning to the same points
   - Unnecessary elaboration after the answer is clear

### For Open-Ended Problems (essays, explanations, creative tasks, discussions):
No overthinking score is calculated, but still detect:
- **Redundant sections**: Sections that repeat previous content
- **Unnecessary elaboration**: Over-explanation of simple points
- **Circular reasoning**: Going in circles without adding value

## Output Format

You MUST respond with valid JSON in this EXACT format:

```json
{
  "batch_summary": "Brief summary of what these sections accomplish",
  "claims": [
    {
      "section_id": 1,
      "claim": "The problem requires finding the area of a circle with radius 5"
    }
  ],
  "answer_conclusions": [
    {
      "section_id": 3,
      "answer": "The area is 78.54 square units",
      "is_final": true,
      "changes_previous_answer": false,
      "quote": "Therefore, the final answer is 78.54 square units"
    }
  ],
  "reflections": [
    {
      "section_id": 5,
      "previous_answer": "25π",
      "new_answer": "78.54",
      "is_correction": true,
      "is_effective": true,
      "quote": "Wait, I need to compute the numerical value: 25π ≈ 78.54"
    }
  ],
  "redundant_sections": [
    {
      "section_id": 8,
      "redundancy_type": "repetitive_verification",
      "reason": "Section 8 repeats the same verification already done in section 6",
      "quote": "Let me verify once more..."
    }
  ],
  "errors": [
    {
      "type": "formal_error",
      "section_ids": [2],
      "reason": "Section 2 states 10 + 2 = 11, which is incorrect (should be 12)"
    }
  ]
}
```

## Field Descriptions

### claims (REQUIRED)
Extract the key claim or conclusion from each section:
- `section_id`: The section number
- `claim`: A concise summary of what this section claims or concludes

### answer_conclusions (include when a section provides an answer)
Track sections that provide answer conclusions:
- `section_id`: The section number containing the answer
- `answer`: The answer provided
- `is_final`: Whether this appears to be the final/complete answer
- `changes_previous_answer`: Whether this changes a previously stated answer
- `quote`: The exact quote from the section

### reflections (include when the model reconsiders its answer)
Track when the model changes its answer:
- `section_id`: Where the reflection occurs
- `previous_answer`: The answer before this reflection
- `new_answer`: The answer after this reflection
- `is_correction`: Whether this fixes a mistake (true) or is just rephrasing (false)
- `is_effective`: Whether this reflection moves toward the correct answer
- `quote`: The relevant quote showing the reflection

### redundant_sections (include when repetitive/unnecessary reasoning is detected)
Identify sections with redundant content:
- `section_id`: The redundant section
- `redundancy_type`: One of "repetitive_verification", "circular_reasoning", "unnecessary_elaboration"
- `reason`: Why this section is redundant
- `quote`: The relevant quote

### errors (include only if errors are found)
List each distinct error:
- `type`: One of "safety_issue", "knowledge_error", "logical_error", "formal_error", "hallucination", "readability_issue"
- `section_ids`: List of section IDs where this error occurs
- `reason`: Explanation of the error

## Important Guidelines

### Answer Tracking
- Track the FIRST answer conclusion as the initial answer
- Note when answers change (reflections)
- Compare with the expected answer to determine if reflections are effective
- For open-ended problems, focus on redundancy detection rather than answer tracking

### Redundancy Detection
- Look for sections that repeat what was already established
- Identify unnecessary verification after conclusions are clear
- Note circular reasoning that doesn't advance understanding
- Flag over-elaboration on simple points

### Formal Errors (Calculations)
- Use the calculator tool to verify complex calculations
- SKIP trivial calculations (e.g., 1+1, 2+2, 10-5)
- Only verify calculations that are critical to the reasoning

### Multiple Errors
- Each distinct error should be a separate entry
- Do NOT combine unrelated errors of the same type

## Calculator Tool

For calculations you need to verify, use:
```
calculator(expression="10 + 2", expected_result="11")
```

Supports: +, -, *, /, ^ (power), sqrt(), abs(), sin(), cos(), tan(), log(), exp()
"""


@dataclass
class SectionAnalysisResult:
    """Result from analyzing a batch of sections."""

    # Rolling summary for memory propagation
    batch_summary: str

    # Key claims extracted from each section
    claims: List[Dict[str, Any]] = field(default_factory=list)

    # === Answer Evolution Tracking ===

    # Answer conclusions found in this batch
    # Each: {section_id, answer, is_final, quote, changes_previous_answer}
    answer_conclusions: List[Dict[str, Any]] = field(default_factory=list)

    # Reflections detected in this batch (answer changes/corrections)
    # Each: {section_id, previous_answer, new_answer, is_correction, quote}
    reflections: List[Dict[str, Any]] = field(default_factory=list)

    # === Redundancy Detection ===

    # Redundant sections detected in this batch
    # Each: {section_id, redundancy_type, reason, quote}
    redundant_sections: List[Dict[str, Any]] = field(default_factory=list)

    # === Errors (other types) ===

    # All errors in a unified list format
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Batch metadata
    batch_start_section: int = 0
    batch_end_section: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "batch_summary": self.batch_summary,
            "claims": self.claims,
            "answer_conclusions": self.answer_conclusions,
            "reflections": self.reflections,
            "redundant_sections": self.redundant_sections,
            "errors": self.errors,
            "batch_start_section": self.batch_start_section,
            "batch_end_section": self.batch_end_section,
        }


@dataclass
class OverthinkingAnalysis:
    """Analysis of overthinking patterns in reasoning.

    Supports two types of problems:
    1. Fixed-answer problems: Calculate overthinking score based on answer evolution
    2. Open-ended problems: Only detect redundant/repetitive sections
    """

    # Problem type classification
    is_fixed_answer_problem: bool = False
    problem_type_reason: str = ""  # Why this classification was made

    # === Answer Evolution Tracking (for fixed-answer problems) ===

    # Initial answer (may be wrong)
    initial_answer: Optional[str] = None
    initial_answer_section: Optional[int] = None
    initial_answer_correct: Optional[bool] = None

    # Reflections that changed the answer
    # Each: {section_id, previous_answer, new_answer, is_correction, is_effective, quote}
    reflections: List[Dict[str, Any]] = field(default_factory=list)

    # Final answer info
    final_answer: Optional[str] = None
    final_answer_section: Optional[int] = None
    final_answer_correct: Optional[bool] = None

    # === Redundancy Detection (for both problem types) ===

    # Redundant sections with explanations
    # Each: {section_id, redundancy_type, reason, quote}
    # redundancy_type: "repetitive_verification", "circular_reasoning", "unnecessary_elaboration"
    redundant_sections: List[Dict[str, Any]] = field(default_factory=list)

    # === Metrics ===

    # Total sections analyzed
    total_sections: int = 0

    # Overthinking score (0-1, only meaningful for fixed-answer problems)
    # Calculated based on: ineffective reasoning after correct answer
    overthinking_score: float = 0.0

    # Counts
    effective_reflection_count: int = 0  # Reflections that improved the answer
    ineffective_reasoning_count: int = 0  # Redundant sections after correct answer

    # Detailed breakdown
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_fixed_answer_problem": self.is_fixed_answer_problem,
            "problem_type_reason": self.problem_type_reason,
            "initial_answer": self.initial_answer,
            "initial_answer_section": self.initial_answer_section,
            "initial_answer_correct": self.initial_answer_correct,
            "reflections": self.reflections,
            "final_answer": self.final_answer,
            "final_answer_section": self.final_answer_section,
            "final_answer_correct": self.final_answer_correct,
            "redundant_sections": self.redundant_sections,
            "total_sections": self.total_sections,
            "overthinking_score": self.overthinking_score,
            "effective_reflection_count": self.effective_reflection_count,
            "ineffective_reasoning_count": self.ineffective_reasoning_count,
            "details": self.details,
        }


class SectionAnalysisAgent:
    """
    Agent that analyzes reasoning sections in batches.

    Key features:
    - Processes sections in batches of 10
    - Maintains rolling summary as memory
    - Tracks answer conclusions for overthinking detection
    - Comprehensive safety checking
    """

    DEFAULT_BATCH_SIZE = 10

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.calculator_tool = CalculatorTool()

        # Override batch size to 10 for this agent
        self.batch_size = self.DEFAULT_BATCH_SIZE

        # Memory state - rolling summary from previous batches
        self.accumulated_summary: str = ""

        # All batch results
        self.batch_results: List[SectionAnalysisResult] = []

        # Aggregated findings
        self.all_claims: List[Dict[str, Any]] = []  # Key claims from each section
        self.all_answer_conclusions: List[Dict[str, Any]] = (
            []
        )  # Answer conclusions found
        self.all_reflections: List[Dict[str, Any]] = []  # Answer changes/corrections
        self.all_redundant_sections: List[Dict[str, Any]] = []  # Redundant sections
        self.all_errors: List[Dict[str, Any]] = []  # All errors in unified format

        # Overthinking tracking
        self.overthinking_analysis = OverthinkingAnalysis()

        # Query and final answer for reference
        self.query: str = ""
        self.expected_answer: str = ""
        self.total_sections: int = 0

        # Problem type (determined by first batch or explicitly set)
        self.is_fixed_answer_problem: Optional[bool] = None

        # History tracking for debugging
        self.history: List[Dict[str, Any]] = []

    def reset(self):
        """Reset agent state for new analysis."""
        self.accumulated_summary = ""
        self.batch_results = []
        self.all_claims = []
        self.all_answer_conclusions = []
        self.all_reflections = []
        self.all_redundant_sections = []
        self.all_errors = []
        self.overthinking_analysis = OverthinkingAnalysis()
        self.query = ""
        self.expected_answer = ""
        self.total_sections = 0
        self.is_fixed_answer_problem = None
        self.history = []

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get tool definitions for LLM function calling."""
        return [self.calculator_tool.get_tool_definition()]

    def _get_history_dir(self) -> Path:
        """Get the directory for saving history files.

        If history_output_dir is configured, use that. Otherwise fall back to default.
        """
        # Check if custom output directory is configured
        if self.config.history_output_dir:
            base_dir = Path(self.config.history_output_dir)
        else:
            # Use the default data directory relative to this file
            base_dir = (
                Path(__file__).resolve().parent.parent.parent.parent.parent
                / "data"
                / "error_detection_history"
            )
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def save_history(self, output_dir: Optional[str] = None) -> str:
        """Save history to a JSON file for debugging.

        Args:
            output_dir: Optional output directory. If None, uses default data directory.

        Returns:
            Path to the saved history file.
        """
        if output_dir:
            history_dir = Path(output_dir)
        else:
            history_dir = self._get_history_dir()

        history_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"error_detection_history_{timestamp}.json"
        filepath = history_dir / filename

        # Build comprehensive history data
        history_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.config.model,
                "batch_size": self.batch_size,
                "total_sections": self.total_sections,
                "total_batches": len(self.batch_results),
            },
            "query": self.query,
            "expected_answer": self.expected_answer,
            "is_fixed_answer_problem": self.is_fixed_answer_problem,
            # Memory state at end
            "final_accumulated_summary": self.accumulated_summary,
            # All batch histories with full LLM call details
            "batch_histories": self.history,
            # Aggregated results
            "aggregated_results": {
                "total_claims": len(self.all_claims),
                "claims": self.all_claims,
                "answer_conclusions_count": len(self.all_answer_conclusions),
                "answer_conclusions": self.all_answer_conclusions,
                "reflections_count": len(self.all_reflections),
                "reflections": self.all_reflections,
                "redundant_sections_count": len(self.all_redundant_sections),
                "redundant_sections": self.all_redundant_sections,
                "total_errors": len(self.all_errors),
                "errors": self.all_errors,
            },
            # Overthinking analysis
            "overthinking_analysis": (
                self.overthinking_analysis.to_dict()
                if self.overthinking_analysis
                else None
            ),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False, default=str)

        log.info("Saved error detection history to: %s", filepath)
        return str(filepath)

    async def analyze_all_sections(
        self,
        sections: List[Dict[str, Any]],
        query: str,
        expected_answer: str,
        llm_call_func: Callable,
    ) -> Dict[str, Any]:
        """
        Analyze all sections in batches of 10.

        Args:
            sections: List of section dicts with 'section_number'/'section_id' and 'text'/'content'
            query: The original question/query
            expected_answer: The expected final answer
            llm_call_func: Async function to call LLM

        Returns:
            Comprehensive analysis results including errors and overthinking analysis
        """
        self.reset()

        self.query = query
        self.expected_answer = expected_answer
        self.total_sections = len(sections)
        self.overthinking_analysis.total_sections = len(sections)

        log.info(
            "Starting section analysis: %d sections in batches of %d",
            len(sections),
            self.batch_size,
        )

        # Process sections in batches
        for batch_start in range(0, len(sections), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(sections))
            batch = sections[batch_start:batch_end]

            await self._process_batch(
                batch=batch,
                batch_number=(batch_start // self.batch_size) + 1,
                llm_call_func=llm_call_func,
            )

        # Calculate final overthinking score
        self._calculate_overthinking_analysis()

        # Convert to error objects
        errors = self._build_error_list()

        # Save complete history after analysis
        try:
            history_path = self.save_history()
            log.info("Saved complete error detection history to: %s", history_path)
        except Exception as e:
            log.warning("Failed to save complete history: %s", e)

        return {
            "errors": [e.to_dict() for e in errors],
            "claims": self.all_claims,
            "answer_conclusions": self.all_answer_conclusions,
            "reflections": self.all_reflections,
            "redundant_sections": self.all_redundant_sections,
            "batch_results": [br.to_dict() for br in self.batch_results],
            "overthinking_analysis": self.overthinking_analysis.to_dict(),
            "accumulated_summary": self.accumulated_summary,
            "summary": {
                "total_sections": self.total_sections,
                "total_claims": len(self.all_claims),
                "answer_conclusions_count": len(self.all_answer_conclusions),
                "reflections_count": len(self.all_reflections),
                "redundant_sections_count": len(self.all_redundant_sections),
                "total_errors": len(errors),
                "is_fixed_answer_problem": self.overthinking_analysis.is_fixed_answer_problem,
            },
        }

    async def _process_batch(
        self,
        batch: List[Dict[str, Any]],
        batch_number: int,
        llm_call_func: Callable,
    ):
        """Process a single batch of sections."""
        import time

        batch_start_time = time.time()

        # Get section range
        first_section = batch[0].get("section_number", batch[0].get("section_id", 1))
        last_section = batch[-1].get(
            "section_number", batch[-1].get("section_id", len(batch))
        )

        log.info(
            "Processing batch %d: sections %d-%d",
            batch_number,
            first_section,
            last_section,
        )

        # Build the prompt
        user_message = self._build_batch_prompt(batch, batch_number)

        # Build conversation
        messages = [
            {"role": "system", "content": SECTION_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # Track comprehensive history for this batch
        batch_history = {
            "batch_number": batch_number,
            "sections": f"{first_section}-{last_section}",
            "start_time": datetime.now().isoformat(),
            # Memory state at the start of this batch
            "memory_state_before": {
                "accumulated_summary": self.accumulated_summary,
                "claims_count": len(self.all_claims),
                "answer_conclusions_count": len(self.all_answer_conclusions),
                "reflections_count": len(self.all_reflections),
                "redundant_sections_count": len(self.all_redundant_sections),
                "errors_count": len(self.all_errors),
            },
            # Input sections for this batch
            "input_sections": batch,
            # User prompt (contains memory context)
            "user_prompt": user_message,
            # System prompt (for reference)
            "system_prompt_length": len(SECTION_ANALYSIS_SYSTEM_PROMPT),
            # LLM call iterations
            "llm_iterations": [],
            # Final parsed result
            "parsed_result": None,
            # Timing
            "duration_seconds": None,
            "end_time": None,
        }

        # Run agent loop with tool calling
        tools = self.get_tools()
        max_iterations = 15

        for iteration in range(max_iterations):
            iteration_start_time = time.time()
            iteration_record = {
                "iteration": iteration,
                "start_time": datetime.now().isoformat(),
                "input_messages_count": len(messages),
                "input_messages": messages.copy(),  # Full messages for debugging
                "response": None,
                "response_content": None,
                "tool_calls": [],
                "duration_seconds": None,
                "error": None,
            }

            try:
                log.info(
                    "Batch %d, Iteration %d: Calling LLM with %d messages",
                    batch_number,
                    iteration,
                    len(messages),
                )

                response = await llm_call_func(
                    self.config.model,
                    messages,
                    tools,
                )

                iteration_record["response"] = response
                iteration_record["duration_seconds"] = (
                    time.time() - iteration_start_time
                )

                # Check for tool calls
                tool_calls = self._extract_tool_calls(response)

                if not tool_calls:
                    # No more tool calls, parse final response
                    content = self._extract_content(response)
                    iteration_record["response_content"] = content
                    iteration_record["is_final"] = True

                    # Debug logging: show raw response content
                    if content:
                        log.debug(
                            "LLM response content (first 500 chars): %s",
                            content[:500] if len(content) > 500 else content,
                        )
                    else:
                        log.warning("LLM response content is empty or None")

                    result = self._parse_batch_response(
                        content, first_section, last_section
                    )

                    if result:
                        # Debug logging: show what was parsed
                        log.info(
                            "Batch %d parsed: summary_len=%d, claims=%d, answers=%d, reflections=%d, redundant=%d, errors=%d",
                            batch_number,
                            len(result.batch_summary),
                            len(result.claims),
                            len(result.answer_conclusions),
                            len(result.reflections),
                            len(result.redundant_sections),
                            len(result.errors),
                        )

                        self.batch_results.append(result)
                        batch_history["parsed_result"] = result.to_dict()

                        # Update accumulated summary
                        if self.accumulated_summary:
                            self.accumulated_summary = f"{self.accumulated_summary}\n\n[Sections {first_section}-{last_section}]: {result.batch_summary}"
                        else:
                            self.accumulated_summary = f"[Sections {first_section}-{last_section}]: {result.batch_summary}"

                        # Aggregate findings
                        self.all_claims.extend(result.claims)
                        self.all_answer_conclusions.extend(result.answer_conclusions)
                        self.all_reflections.extend(result.reflections)
                        self.all_redundant_sections.extend(result.redundant_sections)
                        self.all_errors.extend(result.errors)

                    batch_history["llm_iterations"].append(iteration_record)
                    break

                # Has tool calls - record and execute them
                iteration_record["is_final"] = False
                assistant_content = self._extract_content(response)
                iteration_record["response_content"] = assistant_content

                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": tool_calls,
                    }
                )

                for tool_call in tool_calls:
                    tool_result = self._execute_tool_call(tool_call)

                    # Record tool call with full details
                    iteration_record["tool_calls"].append(
                        {
                            "tool_call": tool_call,
                            "result": tool_result.to_dict(),
                        }
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.get("id", f"call_{iteration}"),
                            "name": tool_call.get("function", {}).get(
                                "name", "calculator"
                            ),
                            "content": json.dumps(tool_result.to_dict()),
                        }
                    )

                batch_history["llm_iterations"].append(iteration_record)

            except Exception as e:
                log.warning("Error in batch processing iteration %d: %s", iteration, e)
                import traceback

                iteration_record["error"] = str(e)
                iteration_record["error_traceback"] = traceback.format_exc()
                iteration_record["duration_seconds"] = (
                    time.time() - iteration_start_time
                )
                batch_history["llm_iterations"].append(iteration_record)
                batch_history["error"] = str(e)
                break

        # Record batch completion
        batch_history["end_time"] = datetime.now().isoformat()
        batch_history["duration_seconds"] = time.time() - batch_start_time
        batch_history["total_iterations"] = len(batch_history["llm_iterations"])

        # Memory state after this batch
        batch_history["memory_state_after"] = {
            "accumulated_summary": self.accumulated_summary,
            "claims_count": len(self.all_claims),
            "answer_conclusions_count": len(self.all_answer_conclusions),
            "reflections_count": len(self.all_reflections),
            "redundant_sections_count": len(self.all_redundant_sections),
            "errors_count": len(self.all_errors),
        }

        # Save batch history
        self.history.append(batch_history)

        # Auto-save history after each batch (for debugging timeout issues)
        try:
            self._auto_save_history()
        except Exception as e:
            log.warning("Failed to auto-save history: %s", e)

    def _auto_save_history(self):
        """Auto-save history after each batch for debugging.

        If using custom output directory (analysis_logs), saves as error_detection_progress.json.
        Otherwise, saves to the default error_detection_history directory.
        """
        history_dir = self._get_history_dir()

        # Use different filename based on whether custom directory is configured
        if self.config.history_output_dir:
            # Custom directory (e.g., analysis_logs) - use descriptive filename
            filepath = history_dir / "05_error_detection_progress.json"
        else:
            # Default directory - use fixed filename that gets overwritten
            filepath = history_dir / "error_detection_latest.json"

        history_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.config.model,
                "batch_size": self.batch_size,
                "total_sections": self.total_sections,
                "batches_completed": len(self.batch_results),
                "status": "in_progress",
            },
            "query": self.query,
            "expected_answer": self.expected_answer,
            "accumulated_summary": self.accumulated_summary,
            "batch_histories": self.history,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False, default=str)

    def _build_batch_prompt(
        self, batch: List[Dict[str, Any]], batch_number: int
    ) -> str:
        """Build the prompt for analyzing a batch of sections."""

        # Format sections
        sections_text = []
        for section in batch:
            section_id = section.get("section_number", section.get("section_id", "?"))
            text = section.get("text", section.get("content", ""))
            sections_text.append(f"[Section {section_id}]\n{text}")

        sections_formatted = "\n\n".join(sections_text)

        # Build memory context with claims and answer conclusions from previous batches
        memory_context = ""
        if self.accumulated_summary or self.all_claims or self.all_answer_conclusions:
            memory_parts = []

            if self.accumulated_summary:
                memory_parts.append(
                    f"### Previous Batch Summaries\n{self.accumulated_summary}"
                )

            if self.all_claims:
                # Format claims as a concise list
                claims_text = "\n".join(
                    [
                        f"- Section {c.get('section_id')}: {c.get('claim', '')}"
                        for c in self.all_claims[
                            -20:
                        ]  # Keep last 20 claims to avoid context overflow
                    ]
                )
                memory_parts.append(
                    f"### Key Claims from Previous Sections\n{claims_text}"
                )

            if self.all_answer_conclusions:
                # Format answer conclusions with evolution tracking
                answers_text = "\n".join(
                    [
                        f"- Section {a.get('section_id')}: {a.get('answer', '')} {'[FINAL]' if a.get('is_final') else '[partial]'} {'[CHANGES PREVIOUS]' if a.get('changes_previous_answer') else ''}"
                        for a in self.all_answer_conclusions
                    ]
                )
                memory_parts.append(
                    f"### Answer Conclusions Found So Far\n{answers_text}"
                )

            if self.all_reflections:
                # Format reflections
                reflections_text = "\n".join(
                    [
                        f"- Section {r.get('section_id')}: '{r.get('previous_answer', '')}' → '{r.get('new_answer', '')}' {'[EFFECTIVE]' if r.get('is_effective') else '[INEFFECTIVE]'}"
                        for r in self.all_reflections
                    ]
                )
                memory_parts.append(
                    f"### Reflections (Answer Changes) So Far\n{reflections_text}"
                )

            memory_context = f"""
## Previous Analysis Summary (Memory from earlier sections)

{chr(10).join(memory_parts)}

---
"""

        # Build the full prompt
        prompt = f"""{memory_context}## Original Query
{self.query}

## Expected Final Answer
{self.expected_answer}

## Sections to Analyze (Batch {batch_number})

{sections_formatted}

---

Please analyze these sections and respond with JSON following the specified format.
Remember to:
1. Use the calculator tool to verify any mathematical calculations
2. Track answer evolution: initial answer → reflections (changes) → final answer
3. Detect redundant sections (repetitive verification, circular reasoning, unnecessary elaboration)
4. Identify any errors (safety, knowledge, logical, formal, hallucination, readability)
5. Compare answers with the expected answer to determine if reflections are effective
"""

        return prompt

    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from LLM response."""
        if isinstance(response, dict):
            if "choices" in response:
                choices = response["choices"]
                if choices and len(choices) > 0:
                    message = choices[0].get("message", {})
                    tool_calls = message.get("tool_calls", [])
                    if tool_calls:
                        return tool_calls
            if "tool_calls" in response:
                return response["tool_calls"]
        return []

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> AgentToolResult:
        """Execute a single tool call."""
        function = tool_call.get("function", {})
        name = function.get("name", "")

        try:
            args_str = function.get("arguments", "{}")
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            return AgentToolResult(
                success=False, result=None, error="Failed to parse arguments"
            )

        if name == "calculator":
            result = self.calculator_tool.execute(**args)
            log.debug(
                "Calculator: %s = %s",
                args.get("expression"),
                result.metadata.get("result"),
            )
            return result

        return AgentToolResult(
            success=False, result=None, error=f"Unknown tool: {name}"
        )

    def _extract_content(self, response: Any) -> Optional[str]:
        """Extract text content from LLM response."""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            if "choices" in response:
                choices = response["choices"]
                if choices and len(choices) > 0:
                    return choices[0].get("message", {}).get("content")
            if "content" in response:
                return response["content"]
        return None

    def _parse_batch_response(
        self,
        content: Optional[str],
        batch_start: int,
        batch_end: int,
    ) -> Optional[SectionAnalysisResult]:
        """Parse the agent's response into a SectionAnalysisResult."""
        if not content:
            log.warning("_parse_batch_response: content is empty")
            return None

        original_content = content  # Keep original for error logging

        try:
            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1]

            data = json.loads(content.strip())

            # Log what we successfully parsed
            log.debug("Parsed JSON with keys: %s", list(data.keys()))

        except json.JSONDecodeError as e:
            log.warning("Failed to parse batch response as JSON: %s", e)
            log.debug(
                "Raw content that failed to parse: %s",
                (
                    original_content[:1000]
                    if len(original_content) > 1000
                    else original_content
                ),
            )
            return SectionAnalysisResult(
                batch_summary="[Parse error - could not extract summary]",
                batch_start_section=batch_start,
                batch_end_section=batch_end,
            )

        # Extract batch summary
        batch_summary = data.get("batch_summary", "")

        # Extract claims - list of {section_id, claim}
        claims = data.get("claims", [])

        # Extract answer conclusions - list of {section_id, answer, is_final, changes_previous_answer, quote}
        answer_conclusions = data.get("answer_conclusions", [])

        # Extract reflections - list of {section_id, previous_answer, new_answer, is_correction, is_effective, quote}
        reflections = data.get("reflections", [])

        # Extract redundant sections - list of {section_id, redundancy_type, reason, quote}
        redundant_sections = data.get("redundant_sections", [])

        # Extract errors - list of {type, section_ids, reason}
        errors = data.get("errors", [])

        return SectionAnalysisResult(
            batch_summary=batch_summary,
            claims=claims,
            answer_conclusions=answer_conclusions,
            reflections=reflections,
            redundant_sections=redundant_sections,
            errors=errors,
            batch_start_section=batch_start,
            batch_end_section=batch_end,
        )

    def _calculate_overthinking_analysis(self):
        """
        Calculate overthinking metrics based on answer evolution and redundancy.

        For fixed-answer problems:
        - Track: initial answer → reflections → final answer
        - Calculate overthinking score based on ineffective reasoning after correct answer

        For open-ended problems:
        - Only track redundant sections, no overthinking score
        """

        # Determine problem type based on expected answer
        self._determine_problem_type()

        # Collect all redundant sections
        self.overthinking_analysis.redundant_sections = (
            self.all_redundant_sections.copy()
        )
        self.overthinking_analysis.ineffective_reasoning_count = len(
            self.all_redundant_sections
        )

        if not self.all_answer_conclusions:
            # No answer conclusions found
            self.overthinking_analysis.overthinking_score = 0.0
            return

        # Sort answer conclusions by section_id
        sorted_conclusions = sorted(
            self.all_answer_conclusions, key=lambda x: x.get("section_id", 0)
        )

        # Track initial and final answers
        if sorted_conclusions:
            first_conclusion = sorted_conclusions[0]
            self.overthinking_analysis.initial_answer = first_conclusion.get("answer")
            self.overthinking_analysis.initial_answer_section = first_conclusion.get(
                "section_id"
            )

            # Check if initial answer matches expected
            if self.expected_answer:
                self.overthinking_analysis.initial_answer_correct = self._answers_match(
                    first_conclusion.get("answer", ""), self.expected_answer
                )

            # Find final answer (last one marked as final, or just the last one)
            final_conclusions = [c for c in sorted_conclusions if c.get("is_final")]
            if final_conclusions:
                last_conclusion = final_conclusions[-1]
            else:
                last_conclusion = sorted_conclusions[-1]

            self.overthinking_analysis.final_answer = last_conclusion.get("answer")
            self.overthinking_analysis.final_answer_section = last_conclusion.get(
                "section_id"
            )

            if self.expected_answer:
                self.overthinking_analysis.final_answer_correct = self._answers_match(
                    last_conclusion.get("answer", ""), self.expected_answer
                )

        # Process reflections
        self.overthinking_analysis.reflections = self.all_reflections.copy()
        effective_count = sum(1 for r in self.all_reflections if r.get("is_effective"))
        self.overthinking_analysis.effective_reflection_count = effective_count

        # Calculate overthinking score only for fixed-answer problems
        if self.overthinking_analysis.is_fixed_answer_problem:
            self._calculate_fixed_answer_overthinking_score()
        else:
            # For open-ended problems, no overthinking score
            self.overthinking_analysis.overthinking_score = 0.0
            self.overthinking_analysis.problem_type_reason = (
                "Open-ended problem - overthinking score not applicable. "
                f"Found {len(self.all_redundant_sections)} redundant sections."
            )

        log.info(
            "Overthinking analysis: is_fixed=%s, score=%.2f, initial_section=%s, redundant=%d, reflections=%d",
            self.overthinking_analysis.is_fixed_answer_problem,
            self.overthinking_analysis.overthinking_score,
            self.overthinking_analysis.initial_answer_section,
            len(self.all_redundant_sections),
            len(self.all_reflections),
        )

    def _determine_problem_type(self):
        """Determine if this is a fixed-answer or open-ended problem."""

        # If explicitly set, use that
        if self.is_fixed_answer_problem is not None:
            self.overthinking_analysis.is_fixed_answer_problem = (
                self.is_fixed_answer_problem
            )
            self.overthinking_analysis.problem_type_reason = "Explicitly set by caller"
            return

        # Heuristics to determine problem type
        expected = self.expected_answer.strip().lower() if self.expected_answer else ""

        # If expected answer is short and looks like a number/formula/short answer
        is_fixed = False
        reason = ""

        if not expected:
            is_fixed = False
            reason = "No expected answer provided"
        elif len(expected) < 100:
            # Short expected answers suggest fixed-answer problems
            # Check if it looks like a number, formula, or short definitive answer
            import re

            if re.match(r"^[\d\.\-\+\*/\^π\s\(\)]+$", expected):
                is_fixed = True
                reason = "Expected answer appears to be a numerical value"
            elif re.match(r"^(yes|no|true|false|correct|incorrect)$", expected):
                is_fixed = True
                reason = "Expected answer is a boolean/yes-no response"
            elif len(expected.split()) <= 10:
                is_fixed = True
                reason = "Expected answer is a short definitive response"
            else:
                is_fixed = False
                reason = "Expected answer is a longer text, likely open-ended"
        else:
            is_fixed = False
            reason = "Expected answer is a long text (>100 chars), likely open-ended"

        self.overthinking_analysis.is_fixed_answer_problem = is_fixed
        self.overthinking_analysis.problem_type_reason = reason

    def _answers_match(self, answer1: str, answer2: str) -> bool:
        """Check if two answers match (with some flexibility)."""
        if not answer1 or not answer2:
            return False

        # Normalize both answers
        a1 = answer1.strip().lower()
        a2 = answer2.strip().lower()

        # Direct match
        if a1 == a2:
            return True

        # Check if one contains the other (for partial matches)
        if a1 in a2 or a2 in a1:
            return True

        # Try to extract numerical values for comparison
        import re

        nums1 = re.findall(r"[\d\.]+", a1)
        nums2 = re.findall(r"[\d\.]+", a2)

        if nums1 and nums2:
            try:
                # Compare primary numerical values
                n1 = float(nums1[0])
                n2 = float(nums2[0])
                if abs(n1 - n2) < 0.01:  # Close enough for floating point
                    return True
            except ValueError:
                pass

        return False

    def _calculate_fixed_answer_overthinking_score(self):
        """Calculate overthinking score for fixed-answer problems."""

        # Find the section where the correct answer first appeared
        correct_answer_section = None

        # Check answer conclusions for correctness
        for conclusion in sorted(
            self.all_answer_conclusions, key=lambda x: x.get("section_id", 0)
        ):
            answer = conclusion.get("answer", "")
            if self._answers_match(answer, self.expected_answer):
                correct_answer_section = conclusion.get("section_id")
                break

        if correct_answer_section is None:
            # No correct answer found, can't calculate meaningful score
            self.overthinking_analysis.overthinking_score = 0.0
            self.overthinking_analysis.problem_type_reason = (
                "Fixed-answer problem, but correct answer not found in reasoning"
            )
            return

        # Calculate score based on:
        # 1. Sections after correct answer (potential overthinking)
        # 2. Number of redundant sections
        # 3. Ineffective reflections after correct answer

        sections_after_correct = self.total_sections - correct_answer_section

        if self.total_sections <= 0:
            self.overthinking_analysis.overthinking_score = 0.0
            return

        # Base score: ratio of sections after correct answer
        base_score = (
            sections_after_correct / self.total_sections
            if sections_after_correct > 0
            else 0
        )

        # Count redundant sections after correct answer
        redundant_after_correct = sum(
            1
            for r in self.all_redundant_sections
            if r.get("section_id", 0) > correct_answer_section
        )

        # Count ineffective reflections (answer changes that don't improve)
        ineffective_reflections = sum(
            1
            for r in self.all_reflections
            if not r.get("is_effective", True)
            and r.get("section_id", 0) > correct_answer_section
        )

        # Penalty for redundant reasoning after correct answer
        redundancy_penalty = redundant_after_correct * 0.1

        # Penalty for ineffective reflections
        reflection_penalty = ineffective_reflections * 0.15

        # Combined score (capped at 1.0)
        score = min(1.0, base_score + redundancy_penalty + reflection_penalty)

        self.overthinking_analysis.overthinking_score = score
        self.overthinking_analysis.problem_type_reason = (
            f"Fixed-answer problem. Correct answer first appeared at section {correct_answer_section}. "
            f"Sections after: {sections_after_correct}, Redundant after: {redundant_after_correct}, "
            f"Ineffective reflections: {ineffective_reflections}"
        )

        # Add details
        self.overthinking_analysis.details = {
            "correct_answer_section": correct_answer_section,
            "sections_after_correct": sections_after_correct,
            "redundant_after_correct": redundant_after_correct,
            "ineffective_reflections_after_correct": ineffective_reflections,
            "base_score": base_score,
            "redundancy_penalty": redundancy_penalty,
            "reflection_penalty": reflection_penalty,
            "answer_conclusions": self.all_answer_conclusions,
            "reflections": self.all_reflections,
        }

    def _build_error_list(self) -> List[AgentDetectedError]:
        """Convert all findings to AgentDetectedError objects."""
        errors = []

        # Type to enum mapping
        type_mapping = {
            "safety_issue": (AgentErrorType.SAFETY, AgentErrorSeverity.HIGH),
            "knowledge_error": (
                AgentErrorType.KNOWLEDGE_ERROR,
                AgentErrorSeverity.MEDIUM,
            ),
            "logical_error": (AgentErrorType.LOGICAL_ERROR, AgentErrorSeverity.MEDIUM),
            "formal_error": (AgentErrorType.FORMAL_ERROR, AgentErrorSeverity.MEDIUM),
            "hallucination": (AgentErrorType.HALLUCINATION, AgentErrorSeverity.MEDIUM),
            "readability_issue": (AgentErrorType.READABILITY, AgentErrorSeverity.LOW),
        }

        for error in self.all_errors:
            error_type_str = error.get("type", "")
            section_ids = error.get("section_ids", [])
            reason = error.get("reason", "")

            # Get type and severity from mapping
            type_info = type_mapping.get(error_type_str)
            if type_info and section_ids:
                error_type, severity = type_info
                errors.append(
                    AgentDetectedError(
                        type=error_type,
                        description=reason,
                        severity=severity,
                        section_numbers=section_ids,
                        details={"reason": reason},
                    )
                )

        # Add redundant sections as errors (for both problem types)
        if self.all_redundant_sections:
            # Group redundant sections by type
            redundancy_type_map = {}
            for rs in self.all_redundant_sections:
                r_type = rs.get("redundancy_type", "unknown")
                if r_type not in redundancy_type_map:
                    redundancy_type_map[r_type] = []
                redundancy_type_map[r_type].append(rs)

            for r_type, sections in redundancy_type_map.items():
                section_ids = [
                    s.get("section_id") for s in sections if s.get("section_id")
                ]
                reasons = [s.get("reason", "") for s in sections]

                # Determine severity based on count
                if len(section_ids) > 5:
                    severity = AgentErrorSeverity.HIGH
                elif len(section_ids) > 2:
                    severity = AgentErrorSeverity.MEDIUM
                else:
                    severity = AgentErrorSeverity.LOW

                description = f"Redundant reasoning detected ({r_type}): {len(section_ids)} sections"
                if reasons:
                    description += f". Examples: {'; '.join(reasons[:3])}"

                errors.append(
                    AgentDetectedError(
                        type=AgentErrorType.OVERTHINKING,
                        description=description,
                        severity=severity,
                        section_numbers=section_ids,
                        details={
                            "redundancy_type": r_type,
                            "sections": sections,
                        },
                    )
                )

        # Add overthinking as a standard error only for fixed-answer problems with significant score
        if (
            self.overthinking_analysis.is_fixed_answer_problem
            and self.overthinking_analysis.overthinking_score > 0.1
        ):

            # Determine severity based on score
            if self.overthinking_analysis.overthinking_score > 0.5:
                severity = AgentErrorSeverity.HIGH
            elif self.overthinking_analysis.overthinking_score > 0.3:
                severity = AgentErrorSeverity.MEDIUM
            else:
                severity = AgentErrorSeverity.LOW

            # Build description
            score_percent = int(self.overthinking_analysis.overthinking_score * 100)
            initial_section = self.overthinking_analysis.initial_answer_section
            final_section = self.overthinking_analysis.final_answer_section

            description_parts = [f"Overthinking score: {score_percent}%"]

            if self.overthinking_analysis.initial_answer_correct:
                description_parts.append(
                    f"Correct answer appeared at section {initial_section}, "
                    f"but reasoning continued until section {self.total_sections}"
                )
            elif self.overthinking_analysis.final_answer_correct:
                description_parts.append(
                    f"Initial answer (section {initial_section}) was incorrect, "
                    f"corrected at section {final_section}"
                )

            if self.overthinking_analysis.effective_reflection_count > 0:
                description_parts.append(
                    f"{self.overthinking_analysis.effective_reflection_count} effective reflection(s)"
                )

            if self.overthinking_analysis.ineffective_reasoning_count > 0:
                description_parts.append(
                    f"{self.overthinking_analysis.ineffective_reasoning_count} redundant section(s)"
                )

            description = ". ".join(description_parts)

            # Collect all answer-related section IDs
            answer_section_ids = [
                c.get("section_id")
                for c in self.all_answer_conclusions
                if c.get("section_id")
            ]

            errors.append(
                AgentDetectedError(
                    type=AgentErrorType.OVERTHINKING,
                    description=description,
                    severity=severity,
                    section_numbers=answer_section_ids if answer_section_ids else [1],
                    details={
                        "overthinking_score": self.overthinking_analysis.overthinking_score,
                        "is_fixed_answer_problem": True,
                        "initial_answer": self.overthinking_analysis.initial_answer,
                        "initial_answer_section": self.overthinking_analysis.initial_answer_section,
                        "initial_answer_correct": self.overthinking_analysis.initial_answer_correct,
                        "final_answer": self.overthinking_analysis.final_answer,
                        "final_answer_section": self.overthinking_analysis.final_answer_section,
                        "final_answer_correct": self.overthinking_analysis.final_answer_correct,
                        "effective_reflection_count": self.overthinking_analysis.effective_reflection_count,
                        "ineffective_reasoning_count": self.overthinking_analysis.ineffective_reasoning_count,
                        "reflections": self.overthinking_analysis.reflections,
                        "problem_type_reason": self.overthinking_analysis.problem_type_reason,
                    },
                )
            )

        return errors

    def get_overthinking_score(self) -> float:
        """Get the calculated overthinking score (only meaningful for fixed-answer problems)."""
        return self.overthinking_analysis.overthinking_score

    def get_initial_answer_section(self) -> Optional[int]:
        """Get the section where the initial answer appeared."""
        return self.overthinking_analysis.initial_answer_section

    def get_redundant_sections(self) -> List[Dict[str, Any]]:
        """Get list of redundant sections."""
        return self.overthinking_analysis.redundant_sections

    def get_reflections(self) -> List[Dict[str, Any]]:
        """Get list of reflections (answer changes)."""
        return self.overthinking_analysis.reflections

    def is_fixed_answer_problem(self) -> bool:
        """Check if this is a fixed-answer problem."""
        return self.overthinking_analysis.is_fixed_answer_problem

    def set_problem_type(self, is_fixed: bool, reason: str = ""):
        """Explicitly set the problem type."""
        self.is_fixed_answer_problem = is_fixed
        self.overthinking_analysis.is_fixed_answer_problem = is_fixed
        self.overthinking_analysis.problem_type_reason = reason or (
            "Fixed-answer problem" if is_fixed else "Open-ended problem"
        )

    def get_accumulated_summary(self) -> str:
        """Get the accumulated summary across all batches."""
        return self.accumulated_summary

    async def analyze_all_sections_streaming(
        self,
        sections: List[Dict[str, Any]],
        query: str,
        expected_answer: str,
        llm_call_func: Callable,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming version of analyze_all_sections.

        Yields progress events after each batch completes, allowing the frontend
        to show incremental results and avoid timeout issues.

        Yields:
            - {"type": "start", "total_batches": N, "total_sections": M}
            - {"type": "batch_start", "batch": N, "sections": "X-Y"}
            - {"type": "batch_progress", "batch": N, "message": "..."}
            - {"type": "batch_complete", "batch": N, "result": {...}}
            - {"type": "complete", "result": {...}}
            - {"type": "error", "error": "..."}

        Args:
            sections: List of section dicts with 'section_number'/'section_id' and 'text'/'content'
            query: The original question/query
            expected_answer: The expected final answer
            llm_call_func: Async function to call LLM

        Yields:
            Progress events as dictionaries
        """
        self.reset()

        self.query = query
        self.expected_answer = expected_answer
        self.total_sections = len(sections)
        self.overthinking_analysis.total_sections = len(sections)

        total_batches = (len(sections) + self.batch_size - 1) // self.batch_size

        yield {
            "type": "start",
            "total_batches": total_batches,
            "total_sections": len(sections),
            "batch_size": self.batch_size,
        }

        log.info(
            "Starting streaming section analysis: %d sections in %d batches",
            len(sections),
            total_batches,
        )

        # Process sections in batches, yielding after each
        for batch_idx, batch_start in enumerate(
            range(0, len(sections), self.batch_size)
        ):
            batch_end = min(batch_start + self.batch_size, len(sections))
            batch = sections[batch_start:batch_end]
            batch_number = batch_idx + 1

            first_section = batch[0].get(
                "section_number", batch[0].get("section_id", batch_start + 1)
            )
            last_section = batch[-1].get(
                "section_number", batch[-1].get("section_id", batch_end)
            )

            yield {
                "type": "batch_start",
                "batch": batch_number,
                "total_batches": total_batches,
                "sections": f"{first_section}-{last_section}",
            }

            try:
                # Process this batch
                await self._process_batch(
                    batch=batch,
                    batch_number=batch_number,
                    llm_call_func=llm_call_func,
                )

                # Get the latest batch result
                if self.batch_results and len(self.batch_results) >= batch_number:
                    latest_result = self.batch_results[-1]
                    yield {
                        "type": "batch_complete",
                        "batch": batch_number,
                        "total_batches": total_batches,
                        "sections": f"{first_section}-{last_section}",
                        "result": latest_result.to_dict(),
                        "accumulated_errors": len(self.all_errors),
                        "accumulated_claims": len(self.all_claims),
                        "accumulated_answers": len(self.all_answer_conclusions),
                    }
                else:
                    yield {
                        "type": "batch_complete",
                        "batch": batch_number,
                        "total_batches": total_batches,
                        "sections": f"{first_section}-{last_section}",
                        "result": None,
                        "warning": "No result produced for this batch",
                    }

            except Exception as e:
                log.warning("Error processing batch %d: %s", batch_number, e)
                yield {
                    "type": "batch_error",
                    "batch": batch_number,
                    "total_batches": total_batches,
                    "sections": f"{first_section}-{last_section}",
                    "error": str(e),
                }

        # Calculate final overthinking score
        self._calculate_overthinking_analysis()

        # Convert to error objects
        errors = self._build_error_list()

        # Save complete history after analysis
        history_path = None
        try:
            history_path = self.save_history()
            log.info("Saved complete error detection history to: %s", history_path)
        except Exception as e:
            log.warning("Failed to save complete history: %s", e)

        # Yield final complete result
        final_result = {
            "errors": [e.to_dict() for e in errors],
            "claims": self.all_claims,
            "answer_conclusions": self.all_answer_conclusions,
            "reflections": self.all_reflections,
            "redundant_sections": self.all_redundant_sections,
            "batch_results": [br.to_dict() for br in self.batch_results],
            "overthinking_analysis": self.overthinking_analysis.to_dict(),
            "accumulated_summary": self.accumulated_summary,
            "summary": {
                "total_sections": self.total_sections,
                "total_claims": len(self.all_claims),
                "answer_conclusions_count": len(self.all_answer_conclusions),
                "reflections_count": len(self.all_reflections),
                "redundant_sections_count": len(self.all_redundant_sections),
                "total_errors": len(errors),
                "is_fixed_answer_problem": self.overthinking_analysis.is_fixed_answer_problem,
            },
            "history_file": history_path,
        }

        yield {
            "type": "complete",
            "result": final_result,
        }
