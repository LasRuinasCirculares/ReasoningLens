"""
Agent Base Module

Provides base classes and types for the agentic error detection system.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

log = logging.getLogger(__name__)


class AgentErrorType(str, Enum):
    """Types of errors the agent can detect.

    Aligned with dataset error types for consistency.
    """

    # Core reasoning errors
    OVERTHINKING = "Overthinking"
    SAFETY = "Safety"
    KNOWLEDGE_ERROR = "Knowledge Error"
    LOGICAL_ERROR = "Logical Error"
    FORMAL_ERROR = "Formal Error"  # Calculation/computation errors
    HALLUCINATION = "Hallucination"
    READABILITY = "Readability"

    # Legacy aliases (for backward compatibility)
    @classmethod
    def from_string(cls, value: str) -> "AgentErrorType":
        """Convert string to error type with backward compatibility."""
        # Normalize input
        normalized = value.strip().lower().replace("_", " ").replace("-", " ")

        # Map legacy names to new types
        mappings = {
            "calculation error": cls.FORMAL_ERROR,
            "format error": cls.READABILITY,
            "safety issue": cls.SAFETY,
            "faithfulness issue": cls.HALLUCINATION,
            "faithfulness": cls.HALLUCINATION,
        }

        if normalized in mappings:
            return mappings[normalized]

        # Try direct match
        for member in cls:
            if member.value.lower() == normalized:
                return member

        # Default to logical error
        return cls.LOGICAL_ERROR


class AgentErrorSeverity(str, Enum):
    """Severity levels for agent-detected errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentConfig:
    """Configuration for the error detection agent."""

    # Batch size: number of sections to process at once (default 10 for section agent)
    batch_size: int = 10
    # Model to use for LLM calls
    model: str = "gpt-4"
    # Maximum concurrent LLM calls
    max_concurrent_calls: int = 3
    # Timeout for each LLM call in seconds
    call_timeout: float = 60.0
    # Whether to use calculator tool for verification
    enable_calculator_tool: bool = True
    # Optional directory to save history/debug files (if None, uses default error_detection_history)
    history_output_dir: Optional[str] = None


@dataclass
class AgentToolResult:
    """Result from an agent tool execution."""

    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class AgentDetectedError:
    """Represents an error detected by the agent."""

    type: AgentErrorType
    description: str
    severity: AgentErrorSeverity
    section_numbers: List[int]  # Which sections this error spans
    position: Optional[int] = None
    context: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    tool_results: List[AgentToolResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value,
            "description": self.description,
            "severity": self.severity.value,
            "section_numbers": self.section_numbers,
            "position": self.position,
            "context": self.context,
            "details": self.details,
            "tool_results": [tr.to_dict() for tr in self.tool_results],
        }


class AgentTool(ABC):
    """Abstract base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of what this tool does."""
        pass

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for tool parameters."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> AgentToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            AgentToolResult with execution outcome
        """
        pass

    def get_tool_definition(self) -> Dict[str, Any]:
        """Get tool definition for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }


# Type alias for LLM call function
LLMCallFunc = Callable[[str, List[Dict[str, str]], Optional[List[Dict]]], Any]
