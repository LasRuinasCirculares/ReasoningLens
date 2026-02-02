"""
Reasoning Analysis Models Module

This module contains Pydantic model definitions for the reasoning analysis system.
"""

from typing import Optional
from pydantic import BaseModel


class ReasoningAnalysisRequest(BaseModel):
    """Request model for reasoning analysis endpoint."""

    chat_id: str
    message_id: str
    model: str
    stream: bool = False
    prompt_override: Optional[str] = None
    force: bool = False  # Force re-analysis, ignoring cache
