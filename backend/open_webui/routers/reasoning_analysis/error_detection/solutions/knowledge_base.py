"""
Error Solutions Knowledge Base Manager

Provides functionality to load, query, and manage the error solutions knowledge base.
The knowledge base contains training methods and quick fixes for different error types.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)

# Path to the knowledge base JSON file
DATA_DIR = Path(__file__).parent.parent / "data"
KNOWLEDGE_BASE_FILE = DATA_DIR / "error_solutions.json"


class ErrorSolutionsKnowledgeBase:
    """
    Manages the error solutions knowledge base.

    Provides methods to:
    - Load and cache the knowledge base
    - Query solutions for specific error types
    - Add new training methods
    - Update the knowledge base
    """

    _instance: Optional["ErrorSolutionsKnowledgeBase"] = None
    _data: Optional[Dict[str, Any]] = None
    _last_loaded: Optional[datetime] = None

    def __new__(cls) -> "ErrorSolutionsKnowledgeBase":
        """Singleton pattern to ensure only one instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the knowledge base."""
        if self._data is None:
            self._load()

    def _load(self) -> None:
        """Load the knowledge base from the JSON file."""
        try:
            if KNOWLEDGE_BASE_FILE.exists():
                with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                self._last_loaded = datetime.now()
                log.info(
                    "Loaded error solutions knowledge base: %s", KNOWLEDGE_BASE_FILE
                )
            else:
                log.warning("Knowledge base file not found: %s", KNOWLEDGE_BASE_FILE)
                self._data = self._get_default_data()
        except Exception as e:
            log.error("Failed to load knowledge base: %s", e)
            self._data = self._get_default_data()

    def _get_default_data(self) -> Dict[str, Any]:
        """Return default minimal data structure."""
        return {
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "error_types": {},
            "difficulty_levels": {
                "beginner": {"label": "Beginner", "color": "#22c55e"},
                "intermediate": {"label": "Intermediate", "color": "#f59e0b"},
                "advanced": {"label": "Advanced", "color": "#ef4444"},
            },
            "categories": {},
        }

    def _save(self) -> bool:
        """Save the knowledge base to the JSON file."""
        try:
            # Ensure data directory exists
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            # Update last_updated timestamp
            self._data["last_updated"] = datetime.now().strftime("%Y-%m-%d")

            with open(KNOWLEDGE_BASE_FILE, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)

            log.info("Saved error solutions knowledge base")
            return True
        except Exception as e:
            log.error("Failed to save knowledge base: %s", e)
            return False

    def reload(self) -> None:
        """Force reload the knowledge base from disk."""
        self._load()

    def get_all_error_types(self) -> List[str]:
        """Get list of all error types in the knowledge base."""
        if not self._data:
            return []
        return list(self._data.get("error_types", {}).keys())

    def get_error_type_info(self, error_type: str) -> Optional[Dict[str, Any]]:
        """Get full information for a specific error type."""
        if not self._data:
            return None

        error_types = self._data.get("error_types", {})
        # Try exact match first
        if error_type in error_types:
            return error_types[error_type]

        # Try case-insensitive match
        for key, value in error_types.items():
            if key.lower() == error_type.lower():
                return value

        return None

    def get_quick_fixes(self, error_type: str) -> List[str]:
        """Get quick fix suggestions for an error type."""
        info = self.get_error_type_info(error_type)
        if not info:
            return []
        return info.get("quick_fixes", [])

    def get_training_methods(
        self,
        error_type: str,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get training methods for an error type.

        Args:
            error_type: The error type to get methods for
            category: Optional filter by category
            difficulty: Optional filter by difficulty level

        Returns:
            List of training method dictionaries
        """
        info = self.get_error_type_info(error_type)
        if not info:
            return []

        methods = info.get("training_methods", [])

        # Apply filters
        if category:
            methods = [
                m for m in methods if m.get("category", "").lower() == category.lower()
            ]

        if difficulty:
            methods = [
                m
                for m in methods
                if m.get("difficulty", "").lower() == difficulty.lower()
            ]

        return methods

    def get_training_method_categories(self, error_type: str) -> List[str]:
        """Get unique categories of training methods for an error type."""
        info = self.get_error_type_info(error_type)
        if not info:
            return []

        methods = info.get("training_methods", [])
        categories = list(set(m.get("category", "Other") for m in methods))
        return sorted(categories)

    def get_evaluation_metrics(self, error_type: str) -> List[str]:
        """Get evaluation metrics for an error type."""
        info = self.get_error_type_info(error_type)
        if not info:
            return []
        return info.get("evaluation_metrics", [])

    def add_training_method(
        self,
        error_type: str,
        method: Dict[str, Any],
    ) -> bool:
        """
        Add a new training method to an error type.

        Args:
            error_type: The error type to add the method to
            method: Dictionary with method details (name, category, description, etc.)

        Returns:
            True if added successfully, False otherwise
        """
        if not self._data:
            return False

        error_types = self._data.get("error_types", {})

        # Find the error type (case-insensitive)
        target_key = None
        for key in error_types:
            if key.lower() == error_type.lower():
                target_key = key
                break

        if not target_key:
            log.warning("Error type not found: %s", error_type)
            return False

        # Validate method has required fields
        required_fields = ["name", "description"]
        for field in required_fields:
            if field not in method:
                log.warning("Missing required field: %s", field)
                return False

        # Add default values
        method.setdefault("category", "Other")
        method.setdefault("difficulty", "intermediate")
        method.setdefault("effect", "")
        method.setdefault("reference", None)

        # Add to training methods
        if "training_methods" not in error_types[target_key]:
            error_types[target_key]["training_methods"] = []

        error_types[target_key]["training_methods"].append(method)

        return self._save()

    def update_training_method(
        self,
        error_type: str,
        method_name: str,
        updates: Dict[str, Any],
    ) -> bool:
        """
        Update an existing training method.

        Args:
            error_type: The error type containing the method
            method_name: Name of the method to update
            updates: Dictionary of fields to update

        Returns:
            True if updated successfully, False otherwise
        """
        info = self.get_error_type_info(error_type)
        if not info:
            return False

        methods = info.get("training_methods", [])

        for method in methods:
            if method.get("name", "").lower() == method_name.lower():
                method.update(updates)
                return self._save()

        log.warning("Method not found: %s in %s", method_name, error_type)
        return False

    def add_quick_fix(self, error_type: str, quick_fix: str) -> bool:
        """Add a quick fix suggestion to an error type."""
        if not self._data:
            return False

        error_types = self._data.get("error_types", {})

        # Find the error type
        target_key = None
        for key in error_types:
            if key.lower() == error_type.lower():
                target_key = key
                break

        if not target_key:
            return False

        if "quick_fixes" not in error_types[target_key]:
            error_types[target_key]["quick_fixes"] = []

        if quick_fix not in error_types[target_key]["quick_fixes"]:
            error_types[target_key]["quick_fixes"].append(quick_fix)
            return self._save()

        return True  # Already exists

    def get_solutions_summary(self, error_type: str) -> Dict[str, Any]:
        """
        Get a complete summary of solutions for an error type.

        Returns a dictionary suitable for the frontend suggestion window.
        Separates test-time methods (no training required) from training methods.
        """
        info = self.get_error_type_info(error_type)
        if not info:
            return {
                "error_type": error_type,
                "found": False,
                "quick_fixes": [],
                "test_time_methods": [],
                "test_time_methods_by_category": {},
                "test_time_categories": [],
                "training_methods": [],
                "training_methods_by_category": {},
                "categories": [],
                "evaluation_metrics": [],
            }

        # Get test-time methods (no training required)
        test_time_methods = info.get("test_time_methods", [])
        test_time_categories = {}
        for method in test_time_methods:
            cat = method.get("category", "Other")
            if cat not in test_time_categories:
                test_time_categories[cat] = []
            test_time_categories[cat].append(method)

        # Get training methods
        training_methods = info.get("training_methods", [])
        training_categories = {}
        for method in training_methods:
            cat = method.get("category", "Other")
            if cat not in training_categories:
                training_categories[cat] = []
            training_categories[cat].append(method)

        return {
            "error_type": error_type,
            "found": True,
            "display_name": info.get("display_name", error_type),
            "description": info.get("description", ""),
            "severity_default": info.get("severity_default", "medium"),
            "quick_fixes": info.get("quick_fixes", []),
            "test_time_methods": test_time_methods,
            "test_time_methods_by_category": test_time_categories,
            "test_time_categories": list(test_time_categories.keys()),
            "training_methods": training_methods,
            "training_methods_by_category": training_categories,
            "categories": list(training_categories.keys()),
            "evaluation_metrics": info.get("evaluation_metrics", []),
        }

    def get_difficulty_levels(self) -> Dict[str, Dict[str, str]]:
        """Get difficulty level definitions."""
        if not self._data:
            return {}
        return self._data.get("difficulty_levels", {})

    def export_data(self) -> Dict[str, Any]:
        """Export the entire knowledge base."""
        return self._data or {}

    def import_data(self, data: Dict[str, Any], merge: bool = False) -> bool:
        """
        Import data into the knowledge base.

        Args:
            data: Data to import
            merge: If True, merge with existing data; if False, replace

        Returns:
            True if imported successfully
        """
        try:
            if merge and self._data:
                # Merge error types
                existing_types = self._data.get("error_types", {})
                new_types = data.get("error_types", {})

                for error_type, info in new_types.items():
                    if error_type in existing_types:
                        # Merge training methods
                        existing_methods = existing_types[error_type].get(
                            "training_methods", []
                        )
                        new_methods = info.get("training_methods", [])
                        existing_names = {m.get("name") for m in existing_methods}

                        for method in new_methods:
                            if method.get("name") not in existing_names:
                                existing_methods.append(method)

                        existing_types[error_type][
                            "training_methods"
                        ] = existing_methods

                        # Merge quick fixes
                        existing_fixes = existing_types[error_type].get(
                            "quick_fixes", []
                        )
                        new_fixes = info.get("quick_fixes", [])
                        for fix in new_fixes:
                            if fix not in existing_fixes:
                                existing_fixes.append(fix)
                        existing_types[error_type]["quick_fixes"] = existing_fixes
                    else:
                        existing_types[error_type] = info
            else:
                self._data = data

            return self._save()
        except Exception as e:
            log.error("Failed to import data: %s", e)
            return False


# Singleton instance
_kb_instance: Optional[ErrorSolutionsKnowledgeBase] = None


def get_knowledge_base() -> ErrorSolutionsKnowledgeBase:
    """Get the singleton knowledge base instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = ErrorSolutionsKnowledgeBase()
    return _kb_instance


# Convenience functions
def get_solutions_for_error_type(error_type: str) -> Dict[str, Any]:
    """Get complete solutions summary for an error type."""
    return get_knowledge_base().get_solutions_summary(error_type)


def get_all_error_types() -> List[str]:
    """Get list of all error types."""
    return get_knowledge_base().get_all_error_types()


def get_training_methods(
    error_type: str,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get training methods for an error type."""
    return get_knowledge_base().get_training_methods(error_type, category, difficulty)


def add_training_method(error_type: str, method: Dict[str, Any]) -> bool:
    """Add a training method to an error type."""
    return get_knowledge_base().add_training_method(error_type, method)
