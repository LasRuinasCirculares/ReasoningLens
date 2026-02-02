"""
Agentic Error Detection Module

This module implements an agentic LLM-based error detection system for chain-of-thought
reasoning analysis. The agent processes sections in batches and maintains context
through rolling summaries.

## Error Types (aligned with dataset evaluation)

1. **Overthinking**: Excessive reasoning after finding the correct answer
2. **Safety**: Harmful, dangerous, or discriminatory content
3. **Knowledge Error**: Incorrect facts, theorems, or definitions
4. **Logical Error**: Invalid reasoning steps, contradictions
5. **Formal Error**: Calculation/computation mistakes
6. **Hallucination**: Fabricated information, ignoring constraints
7. **Readability**: Poorly structured content, format issues

## Architecture

The system uses:
- `SectionAnalysisAgent`: LLM agent with batch processing
  - Processes sections in batches of 10 (configurable)
  - Uses rolling summaries for memory efficiency
  - Can call calculator tool for mathematical verification
  - Tracks answer conclusions for overthinking detection
  - Detects all 7 error types in a single pass

## Key Features

- **Batch processing**: Efficient analysis of multiple sections at once
- **Rolling summaries**: Memory-efficient context propagation
- **Calculator tool**: Verifies mathematical calculations with step-by-step evaluation
- **Overthinking detection**: Identifies unnecessary reasoning after finding correct answer

## Usage

```python
from .agent import SectionAnalysisAgent, AgentConfig

agent = SectionAnalysisAgent(
    config=AgentConfig(model="gpt-4", batch_size=10)
)

# Run detection on all sections
result = await agent.analyze_all_sections(
    sections=[...],  # List of section dicts
    query="What is 2 + 2?",
    expected_answer="4",
    llm_call_func=your_llm_function,
)
```
"""

from .base import (
    AgentTool,
    AgentToolResult,
    AgentConfig,
    AgentDetectedError,
    AgentErrorType,
    AgentErrorSeverity,
)
from .calculator_tool import (
    CalculatorTool,
    evaluate_expression_for_agent,
)
from .section_analysis_agent import (
    SectionAnalysisAgent,
    SectionAnalysisResult,
    OverthinkingAnalysis,
)

__all__ = [
    # Base classes
    "AgentTool",
    "AgentToolResult",
    "AgentConfig",
    "AgentDetectedError",
    "AgentErrorType",
    "AgentErrorSeverity",
    # Tools
    "CalculatorTool",
    "evaluate_expression_for_agent",
    # Section analysis agent
    "SectionAnalysisAgent",
    "SectionAnalysisResult",
    "OverthinkingAnalysis",
]
