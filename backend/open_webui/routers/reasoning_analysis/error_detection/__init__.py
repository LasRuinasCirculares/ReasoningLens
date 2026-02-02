"""
Error Detection Module

This module provides LLM-based error detection for chain-of-thought reasoning analysis.

## Error Types (aligned with dataset evaluation)

1. **Overthinking**: Excessive reasoning after finding the answer
2. **Safety**: Harmful, dangerous, or discriminatory content
3. **Knowledge Error**: Factually incorrect information
4. **Logical Error**: Flawed reasoning processes
5. **Formal Error**: Calculation/computation mistakes
6. **Hallucination**: Fabricated information
7. **Readability**: Poorly structured content

## Usage

```python
from .error_detection.agent import SectionAnalysisAgent, AgentConfig

agent = SectionAnalysisAgent(config=AgentConfig(batch_size=10, model="gpt-4"))

result = await agent.analyze_all_sections(
    sections=[...],
    query="What is 2 + 2?",
    expected_answer="4",
    llm_call_func=your_llm_function,
)

errors = result.get("errors", [])
overthinking = result.get("overthinking_analysis", {})
```

## Architecture

The error detection system uses:
- `SectionAnalysisAgent`: LLM-based batch analysis with calculator tool support
  - Processes sections in batches of 10
  - Uses rolling summaries for memory efficiency
  - Detects all 7 error types in a single pass
  - Tracks answer conclusions for overthinking detection
"""

# Import agent module - the main interface
from .agent import (
    AgentConfig,
    SectionAnalysisAgent,
    CalculatorTool,
    AgentErrorType,
    AgentErrorSeverity,
    AgentDetectedError,
)


# Export all public interfaces
__all__ = [
    # Agent module - main interface
    "AgentConfig",
    "SectionAnalysisAgent",
    "CalculatorTool",
    "AgentErrorType",
    "AgentErrorSeverity",
    "AgentDetectedError",
]
