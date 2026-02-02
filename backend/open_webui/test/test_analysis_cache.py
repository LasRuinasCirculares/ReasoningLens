#!/usr/bin/env python3
"""
Test module for reasoning analysis cache structure.

This module tests the cache directory structure and retrieval functionality
for the reasoning analysis system.
"""

import json
import pytest
import shutil
from pathlib import Path

from open_webui.routers.reasoning_analysis.utils import (
    get_message_analysis_dir,
    get_cached_analysis,
    persist_analysis_artifact,
)


# Test configuration
LOG_DIR = Path(__file__).parent.parent / "data" / "analysis_logs_test"
TEST_MESSAGE_ID = "test-message-123"
TEST_MODEL = "gpt-4o-mini"


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up test files before and after each test."""
    if LOG_DIR.exists():
        shutil.rmtree(LOG_DIR)
    yield
    if LOG_DIR.exists():
        shutil.rmtree(LOG_DIR)


def test_directory_structure():
    """Test that the directory structure is created correctly."""
    message_dir = get_message_analysis_dir(LOG_DIR, TEST_MESSAGE_ID, TEST_MODEL)

    # Create a test artifact
    test_metadata = {
        "message_id": TEST_MESSAGE_ID,
        "model": TEST_MODEL,
        "chat_id": "test-chat-456",
    }

    test_payload = {
        "test": "data",
        "nodes": [{"id": "node1", "label": "Test Node"}],
        "edges": [],
    }

    # Save test artifact
    persist_analysis_artifact(
        stage="final_merged_result",
        payload=test_payload,
        metadata=test_metadata,
        log_dir=LOG_DIR,
        enabled=True,
    )

    # Verify directory exists
    assert message_dir.exists(), f"Directory {message_dir} was not created"
    assert message_dir.is_dir(), f"{message_dir} is not a directory"

    # Verify file exists
    result_file = message_dir / "final_merged_result.json"
    assert result_file.exists(), f"File {result_file} was not created"


def test_cache_retrieval():
    """Test that cached analysis can be retrieved."""
    # First create a cached analysis
    test_metadata = {
        "message_id": TEST_MESSAGE_ID,
        "model": TEST_MODEL,
        "chat_id": "test-chat-456",
    }

    test_payload = {
        "test": "data",
        "nodes": [{"id": "node1", "label": "Test Node"}],
        "edges": [],
    }

    persist_analysis_artifact(
        stage="final_merged_result",
        payload=test_payload,
        metadata=test_metadata,
        log_dir=LOG_DIR,
        enabled=True,
    )

    # Try to retrieve the cached analysis
    cached = get_cached_analysis(LOG_DIR, TEST_MESSAGE_ID, TEST_MODEL)

    assert cached is not None, "Failed to retrieve cached analysis"
    assert "payload" in cached, "Cached analysis missing payload"
    assert cached["payload"]["test"] == "data", "Cached data mismatch"


def test_different_models():
    """Test that different models create separate directories."""
    test_model_2 = "claude-3-opus"

    message_dir_1 = get_message_analysis_dir(LOG_DIR, TEST_MESSAGE_ID, TEST_MODEL)
    message_dir_2 = get_message_analysis_dir(LOG_DIR, TEST_MESSAGE_ID, test_model_2)

    assert (
        message_dir_1 != message_dir_2
    ), "Different models should have different directories"
