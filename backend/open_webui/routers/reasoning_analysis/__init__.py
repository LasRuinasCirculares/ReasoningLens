"""
Reasoning Analysis Package

This package provides a two-layer reasoning analysis system for analyzing
AI assistant reasoning processes.

## Package Structure

```
reasoning_analysis/
├── __init__.py           # Main entry point, exports public API
├── models.py             # Data models (Pydantic)
├── prompts.py            # LLM prompt templates (bilingual: EN/ZH)
│
├── core/                 # Core analysis logic
│   ├── layer1.py         # Layer 1 (coarse-grained) analysis
│   ├── layer2.py         # Layer 2 (fine-grained) analysis
│   └── result_builder.py # Final result builder
│
├── extractors/           # Text extraction and processing
│   ├── reasoning.py      # Extract reasoning from messages
│   └── sections.py       # Section splitting and processing
│
├── error_detection/      # LLM-based error detection system
│   ├── __init__.py       # Package entry point
│   ├── agent/            # LLM-based detection agent
│   │   ├── base.py           # Agent base classes and types
│   │   ├── calculator_tool.py # Safe calculator for math verification
│   │   └── section_analysis_agent.py # Main batch analysis agent
│   ├── solutions/        # Error solutions knowledge base
│   │   └── knowledge_base.py # Training methods and quick fixes
│   └── data/             # Data files
│       └── error_solutions.json # Knowledge base data
│
└── utils/                # Utilities
    ├── json_utils.py     # JSON parsing utilities
    └── cache.py          # Caching and persistence
```

## Usage

```python
from open_webui.routers.reasoning_analysis import (
    ReasoningAnalysisRequest,
    analyze_layer1,
    analyze_layer2_for_node,
    build_final_result,
    extract_reasoning_and_answer,
)
```
"""

# Models
from .models import ReasoningAnalysisRequest

# Core Analysis
from .core import (
    analyze_layer1,
    analyze_layer2_for_node,
    build_final_result,
)

# Extractors
from .extractors import (
    extract_reasoning_and_answer,
    split_reasoning_into_sections,
    is_continuation_section,
    StreamingReasoningSectionBuilder,
    CONTINUATION_WORDS,
)

# Error Detection - LLM-based with SectionAnalysisAgent
from .error_detection import (
    AgentConfig,
    SectionAnalysisAgent,
    AgentErrorType,
    AgentErrorSeverity,
    AgentDetectedError,
)

__all__ = [
    # Models
    "ReasoningAnalysisRequest",
    # Core Analyzers
    "analyze_layer1",
    "analyze_layer2_for_node",
    "build_final_result",
    # Extractors
    "extract_reasoning_and_answer",
    "split_reasoning_into_sections",
    "is_continuation_section",
    "StreamingReasoningSectionBuilder",
    "CONTINUATION_WORDS",
    # Error Detection
    "AgentConfig",
    "SectionAnalysisAgent",
    "AgentErrorType",
    "AgentErrorSeverity",
    "AgentDetectedError",
]
