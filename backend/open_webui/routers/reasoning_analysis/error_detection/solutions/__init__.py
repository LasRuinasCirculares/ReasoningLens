"""
Error Solutions Knowledge Base Module

This module provides access to the error solutions knowledge base,
which contains optimization methods and suggestions for different error types.
"""

from .knowledge_base import (
    ErrorSolutionsKnowledgeBase,
    get_solutions_for_error_type,
    get_all_error_types,
    get_training_methods,
    add_training_method,
    get_knowledge_base,
)

__all__ = [
    "ErrorSolutionsKnowledgeBase",
    "get_solutions_for_error_type",
    "get_all_error_types",
    "get_training_methods",
    "add_training_method",
    "get_knowledge_base",
]
