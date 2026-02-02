"""
Model Analysis Report Generator

This module generates comprehensive analysis reports for reasoning models
based on aggregated error detection results across multiple conversations.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, AsyncGenerator

from open_webui.models.chats import Chats

log = logging.getLogger(__name__)

# Memory compression thresholds (in estimated tokens)
MEMORY_MAX_TOKENS = 20000  # Compress when memory exceeds this
MEMORY_TARGET_TOKENS = 5000  # Target size after compression


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.

    Uses a simple heuristic: ~4 characters per token for English,
    ~2 characters per token for Chinese.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Count Chinese characters (CJK Unified Ideographs)
    chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_chars = len(text) - chinese_chars

    # Chinese: ~1.5 chars per token, Other: ~4 chars per token
    chinese_tokens = chinese_chars / 1.5
    other_tokens = other_chars / 4

    return int(chinese_tokens + other_tokens)


def get_report_debug_dir() -> Path:
    """Get or create the report generation debug directory."""
    debug_dir = (
        Path(__file__).resolve().parent.parent.parent / "data" / "report_debug_logs"
    )
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def save_llm_call_debug(
    session_id: str,
    step_name: str,
    step_index: int,
    messages: List[Dict[str, str]],
    response_content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save LLM call input/output to a JSON file for debugging.

    Args:
        session_id: Unique session ID for this report generation
        step_name: Name of the step (e.g., "incremental_summary", "compression", "final_report")
        step_index: Index of this step (for ordering)
        messages: The messages sent to the LLM
        response_content: The response content from the LLM
        metadata: Optional additional metadata
    """
    try:
        debug_dir = get_report_debug_dir() / session_id
        debug_dir.mkdir(parents=True, exist_ok=True)

        debug_data = {
            "timestamp": datetime.now().isoformat(),
            "step_name": step_name,
            "step_index": step_index,
            "messages": messages,
            "response": response_content,
            "response_tokens_estimate": estimate_tokens(response_content),
            "metadata": metadata or {},
        }

        # Calculate input tokens
        input_text = "\n".join(m.get("content", "") for m in messages)
        debug_data["input_tokens_estimate"] = estimate_tokens(input_text)

        filename = f"{step_index:02d}_{step_name}.json"
        filepath = debug_dir / filename

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(debug_data, f, ensure_ascii=False, indent=2)

        log.debug("Saved debug log to %s", filepath)
    except Exception as e:
        log.warning("Failed to save debug log: %s", e)


def get_reasoning_model_from_chat(chat_id: str, message_id: str) -> Optional[str]:
    """
    Get the reasoning model (original model that generated the message) from the chat database.

    Args:
        chat_id: The chat ID
        message_id: The message ID

    Returns:
        The model name if found, None otherwise
    """
    try:
        chat = Chats.get_chat_by_id(chat_id)
        if chat is None:
            return None

        messages_map = chat.chat.get("history", {}).get("messages", {}) or {}
        message = messages_map.get(message_id)
        if message:
            return message.get("model")
        return None
    except Exception as e:
        log.debug("Failed to get reasoning model from chat %s: %s", chat_id, e)
        return None


@dataclass
class AnalysisRecord:
    """A single analysis record from history."""

    message_id: str
    chat_id: str
    model: str
    timestamp: str
    # Layer 1 analysis results - core data for report generation
    nodes: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Node descriptions and tree structure
    edges: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Connections between nodes
    sections: List[Dict[str, Any]] = field(default_factory=list)  # Reasoning segments
    # Error detection results - key for understanding model characteristics
    errors: List[Dict[str, Any]] = field(
        default_factory=list
    )  # Error types and descriptions
    overthinking_analysis: Optional[Dict[str, Any]] = None  # Overthinking metrics
    # Original query (for context in report)
    query: str = ""


@dataclass
class AggregatedStats:
    """Aggregated statistics from multiple analysis records."""

    total_analyses: int = 0
    total_errors: int = 0
    total_sections: int = 0

    # Error distribution - key for understanding model error patterns
    error_type_counts: Dict[str, int] = field(default_factory=dict)

    # Node type distribution - for analyzing reasoning structure patterns
    node_type_counts: Dict[str, int] = field(default_factory=dict)

    # Example errors with descriptions (for detailed report generation)
    error_examples: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    # Structured reasoning tree summaries (simplified input for report)
    reasoning_trees: List[Dict[str, Any]] = field(default_factory=list)

    # Overthinking statistics
    avg_overthinking_score: float = 0.0
    high_overthinking_count: int = 0  # score > 0.5
    avg_sections: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_analyses": self.total_analyses,
            "total_errors": self.total_errors,
            "total_sections": self.total_sections,
            "total_nodes": sum(self.node_type_counts.values()),
            "error_type_counts": self.error_type_counts,
            "node_type_counts": self.node_type_counts,
            "avg_overthinking_score": self.avg_overthinking_score,
            "high_overthinking_count": self.high_overthinking_count,
            "avg_sections": self.avg_sections,
            "error_examples": {
                k: v[:3]
                for k, v in self.error_examples.items()  # Limit to 3 examples per type
            },
            "reasoning_trees": self.reasoning_trees[:5],  # Limit to 5 tree examples
        }


def get_analysis_logs_dir() -> Path:
    """Get the analysis logs directory."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "analysis_logs"


# =============================================================================
# Saved Reports Management
# =============================================================================


@dataclass
class SavedReport:
    """A saved report record."""

    report_id: str  # Directory name as ID
    reasoning_model: str
    report_model: str
    analysis_model: str
    timestamp: str
    records_count: int
    report_content: str
    language: str

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary (without full content)."""
        return {
            "report_id": self.report_id,
            "reasoning_model": self.reasoning_model,
            "report_model": self.report_model,
            "analysis_model": self.analysis_model,
            "timestamp": self.timestamp,
            "records_count": self.records_count,
            "content_preview": (
                self.report_content[:200] + "..."
                if len(self.report_content) > 200
                else self.report_content
            ),
            "language": self.language,
        }

    def to_full_dict(self) -> Dict[str, Any]:
        """Convert to full dictionary (with content)."""
        return {
            "report_id": self.report_id,
            "reasoning_model": self.reasoning_model,
            "report_model": self.report_model,
            "analysis_model": self.analysis_model,
            "timestamp": self.timestamp,
            "records_count": self.records_count,
            "report_content": self.report_content,
            "language": self.language,
        }


def find_final_report_file(report_dir: Path) -> Optional[Path]:
    """
    Find the final report file in a report directory.

    The step_index is dynamic based on the number of batches processed,
    so we need to search for files matching *_final_report.json pattern.

    Args:
        report_dir: Path to the report directory

    Returns:
        Path to the final report file if found, None otherwise
    """
    if not report_dir.exists() or not report_dir.is_dir():
        return None

    # Look for files matching pattern: XX_final_report.json (not XX_final_report_input.json)
    for f in report_dir.iterdir():
        if (
            f.is_file()
            and f.name.endswith("_final_report.json")
            and "_input" not in f.name
        ):
            return f

    return None


def list_saved_reports(
    reasoning_model: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    List all saved reports from report_debug_logs directory.

    Args:
        reasoning_model: Optional filter by reasoning model
        limit: Maximum number of reports to return

    Returns:
        List of saved report summaries (without full content)
    """
    debug_dir = get_report_debug_dir()

    if not debug_dir.exists():
        return []

    reports = []

    for report_dir in debug_dir.iterdir():
        if not report_dir.is_dir():
            continue

        # Look for the final report file (dynamic step_index)
        final_report_file = find_final_report_file(report_dir)
        if not final_report_file:
            continue

        try:
            with final_report_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            metadata = data.get("metadata", {})
            report_reasoning_model = metadata.get("reasoning_model", "unknown")

            # Apply filter if specified
            if reasoning_model:
                safe_filter = re.sub(r"[^a-zA-Z0-9_-]+", "_", reasoning_model).lower()
                safe_model = re.sub(
                    r"[^a-zA-Z0-9_-]+", "_", report_reasoning_model
                ).lower()
                if safe_filter not in safe_model:
                    continue

            # Parse directory name for timestamp: {reasoning_model}_{report_model}_{datetime}
            # e.g., zhuque_Qwen3-32B_20260122_021501
            dir_name = report_dir.name
            # Extract datetime from the end (format: YYYYMMDD_HHMMSS)
            parts = dir_name.rsplit("_", 2)
            if len(parts) >= 3:
                timestamp_str = f"{parts[-2]}_{parts[-1]}"
                # Parse and format nicely
                try:
                    dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    timestamp = dt.isoformat()
                except ValueError:
                    timestamp = data.get("timestamp", "unknown")
            else:
                timestamp = data.get("timestamp", "unknown")

            report = SavedReport(
                report_id=report_dir.name,
                reasoning_model=report_reasoning_model,
                report_model=metadata.get("model", "unknown"),
                analysis_model=metadata.get("analysis_model", ""),
                timestamp=timestamp,
                records_count=metadata.get("records_count", 0),
                report_content=data.get("response", ""),
                language=(
                    "zh"
                    if any(
                        "\u4e00" <= c <= "\u9fff"
                        for c in data.get("response", "")[:100]
                    )
                    else "en"
                ),
            )

            reports.append(report.to_summary_dict())

        except Exception as e:
            log.warning("Failed to load report from %s: %s", report_dir, e)
            continue

    # Sort by timestamp descending (newest first)
    reports.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return reports[:limit]


def get_saved_report(report_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific saved report by ID.

    Args:
        report_id: The report directory name

    Returns:
        Full report data or None if not found
    """
    debug_dir = get_report_debug_dir()
    report_dir = debug_dir / report_id

    if not report_dir.exists() or not report_dir.is_dir():
        return None

    # Look for the final report file (dynamic step_index)
    final_report_file = find_final_report_file(report_dir)
    if not final_report_file:
        return None

    try:
        with final_report_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})

        # Parse timestamp from directory name
        dir_name = report_dir.name
        parts = dir_name.rsplit("_", 2)
        if len(parts) >= 3:
            timestamp_str = f"{parts[-2]}_{parts[-1]}"
            try:
                dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                timestamp = dt.isoformat()
            except ValueError:
                timestamp = data.get("timestamp", "unknown")
        else:
            timestamp = data.get("timestamp", "unknown")

        report = SavedReport(
            report_id=report_id,
            reasoning_model=metadata.get("reasoning_model", "unknown"),
            report_model=metadata.get("model", "unknown"),
            analysis_model=metadata.get("analysis_model", ""),
            timestamp=timestamp,
            records_count=metadata.get("records_count", 0),
            report_content=data.get("response", ""),
            language=(
                "zh"
                if any(
                    "\u4e00" <= c <= "\u9fff" for c in data.get("response", "")[:100]
                )
                else "en"
            ),
        )

        return report.to_full_dict()

    except Exception as e:
        log.error("Failed to load report %s: %s", report_id, e)
        return None


def delete_saved_report(report_id: str) -> bool:
    """
    Delete a saved report by ID.

    Args:
        report_id: The report directory name

    Returns:
        True if deleted, False if not found
    """
    import shutil

    debug_dir = get_report_debug_dir()
    report_dir = debug_dir / report_id

    if not report_dir.exists() or not report_dir.is_dir():
        return False

    try:
        shutil.rmtree(report_dir)
        log.info("Deleted saved report: %s", report_id)
        return True
    except Exception as e:
        log.error("Failed to delete report %s: %s", report_id, e)
        return False


def list_analysis_history(
    reasoning_model: Optional[str] = None,
    analysis_model: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    List available analysis history records.

    Args:
        reasoning_model: Filter by reasoning model (the model being analyzed)
        analysis_model: Filter by analysis model (the model used for analysis)
        limit: Maximum number of records to return

    Returns:
        List of analysis history metadata
    """
    log_dir = get_analysis_logs_dir()

    if not log_dir.exists():
        return []

    records = []

    for analysis_dir in log_dir.iterdir():
        if not analysis_dir.is_dir():
            continue

        # Parse directory name: {message_id}_{model}
        dir_name = analysis_dir.name
        parts = dir_name.rsplit("_", 1)
        if len(parts) < 2:
            continue

        # Look for the final merged result file
        result_file = analysis_dir / "final_merged_result.json"
        error_file = analysis_dir / "04_error_detection_result.json"
        input_file = analysis_dir / "00_input.json"

        if not result_file.exists():
            continue

        try:
            # Load the main analysis result
            with result_file.open("r", encoding="utf-8") as f:
                result_data = json.load(f)

            metadata = result_data.get("metadata", {})
            payload = result_data.get("payload", result_data)

            # Get the analysis model from metadata
            analysis_model_used = metadata.get("model", "unknown")

            # Apply analysis model filter if specified
            if analysis_model:
                safe_filter = re.sub(r"[^a-zA-Z0-9_-]+", "_", analysis_model)
                safe_used = re.sub(r"[^a-zA-Z0-9_-]+", "_", analysis_model_used)
                if safe_filter.lower() not in safe_used.lower():
                    continue

            # Load input file to get reasoning model info
            reasoning_model_used = "unknown"
            query_text = ""
            chat_id_from_metadata = metadata.get("chat_id", "unknown")
            message_id_from_metadata = metadata.get("message_id", "unknown")

            if input_file.exists():
                with input_file.open("r", encoding="utf-8") as f:
                    input_data = json.load(f)
                    input_payload = input_data.get("payload", input_data)
                    input_metadata = input_data.get("metadata", {})
                    # Try to get reasoning_model from payload first, then metadata
                    reasoning_model_used = input_payload.get("reasoning_model")
                    if not reasoning_model_used:
                        reasoning_model_used = input_metadata.get("reasoning_model")
                    query_text = input_payload.get("question", "")[:100]

            # Fallback: try to get reasoning model from the chat database
            # This is useful for old records that don't have reasoning_model saved
            if not reasoning_model_used or reasoning_model_used == "unknown":
                db_model = get_reasoning_model_from_chat(
                    chat_id_from_metadata, message_id_from_metadata
                )
                if db_model:
                    reasoning_model_used = db_model
                    log.debug("Got reasoning_model from chat DB: %s", db_model)

            # Last resort: try metadata in result file
            if not reasoning_model_used or reasoning_model_used == "unknown":
                reasoning_model_used = metadata.get("reasoning_model", "unknown")

            # Apply reasoning model filter if specified
            if reasoning_model:
                if reasoning_model.lower() not in reasoning_model_used.lower():
                    continue

            # Get error counts if available
            error_count = 0
            overthinking_score = 0.0
            if error_file.exists():
                try:
                    with error_file.open("r", encoding="utf-8") as f:
                        error_data = json.load(f)
                        error_payload = error_data.get("payload", error_data)
                        error_count = len(error_payload.get("errors", []))
                        ot_analysis = error_payload.get("overthinking_analysis", {})
                        overthinking_score = ot_analysis.get("overthinking_score", 0.0)
                except Exception:
                    pass

            nodes = payload.get("nodes", [])
            sections = payload.get("sections", [])

            record = {
                "message_id": metadata.get("message_id", "unknown"),
                "chat_id": metadata.get("chat_id", "unknown"),
                "analysis_model": analysis_model_used,
                "reasoning_model": reasoning_model_used,
                "query_preview": query_text,
                "node_count": len(nodes),
                "section_count": len(sections),
                "error_count": error_count,
                "overthinking_score": overthinking_score,
                "directory": str(analysis_dir),
                "has_error_detection": error_file.exists(),
            }

            # Try to get file modification time as timestamp
            try:
                mtime = result_file.stat().st_mtime
                record["timestamp"] = datetime.fromtimestamp(mtime).isoformat()
            except Exception:
                record["timestamp"] = "unknown"

            records.append(record)

            if len(records) >= limit:
                break

        except Exception as e:
            log.warning("Failed to load analysis record from %s: %s", analysis_dir, e)
            continue

    # Sort by timestamp (newest first)
    records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return records[:limit]


def load_full_analysis_record(directory: str) -> Optional[AnalysisRecord]:
    """
    Load a full analysis record from a directory.

    Args:
        directory: Path to the analysis directory

    Returns:
        AnalysisRecord with all analysis data
    """
    analysis_dir = Path(directory)

    if not analysis_dir.exists() or not analysis_dir.is_dir():
        return None

    result_file = analysis_dir / "final_merged_result.json"
    error_file = analysis_dir / "04_error_detection_result.json"
    input_file = analysis_dir / "00_input.json"

    if not result_file.exists():
        return None

    try:
        # Load main analysis result
        with result_file.open("r", encoding="utf-8") as f:
            result_data = json.load(f)

        metadata = result_data.get("metadata", {})
        payload = result_data.get("payload", result_data)

        record = AnalysisRecord(
            message_id=metadata.get("message_id", "unknown"),
            chat_id=metadata.get("chat_id", "unknown"),
            model=metadata.get("model", "unknown"),
            timestamp=datetime.now().isoformat(),
            nodes=payload.get("nodes", []),
            edges=payload.get("edges", []),
            sections=payload.get("sections", []),
        )

        # Load input data (only query is needed for report context)
        if input_file.exists():
            with input_file.open("r", encoding="utf-8") as f:
                input_data = json.load(f)
                input_payload = input_data.get("payload", input_data)
                record.query = input_payload.get("question", "")

        # Load error detection results (errors and overthinking analysis)
        if error_file.exists():
            with error_file.open("r", encoding="utf-8") as f:
                error_data = json.load(f)
                error_payload = error_data.get("payload", error_data)
                record.errors = error_payload.get("errors", [])
                record.overthinking_analysis = error_payload.get(
                    "overthinking_analysis"
                )

        return record

    except Exception as e:
        log.error("Failed to load full analysis record from %s: %s", directory, e)
        return None


def extract_simplified_tree(record: AnalysisRecord, index: int) -> Dict[str, Any]:
    """
    Extract simplified reasoning tree structure from an analysis record.

    This focuses on:
    - Node descriptions and types
    - Tree structure (edges)
    - Key reasoning path characteristics

    Args:
        record: The analysis record
        index: Index of the record (for labeling)

    Returns:
        Simplified tree structure dict
    """
    # Extract top-level nodes with descriptions
    simplified_nodes = []
    for node in record.nodes:
        simplified_node = {
            "id": node.get("id", ""),
            "type": node.get("type", "unknown"),
            "label": node.get("label", ""),
            "description": node.get("description", ""),
        }

        # Include Layer 2 tree structure if present (sub-nodes)
        layer2 = node.get("layer2", {})
        if layer2 and "tree" in layer2:
            sub_tree = layer2["tree"]
            sub_nodes = sub_tree.get("nodes", [])
            if sub_nodes:
                simplified_node["sub_nodes"] = [
                    {
                        "type": sn.get("type", "unknown"),
                        "label": sn.get("label", ""),
                        "description": sn.get("description", ""),
                    }
                    for sn in sub_nodes
                ]
                # Include sub-edges to show reasoning flow
                sub_edges = sub_tree.get("edges", [])
                if sub_edges:
                    simplified_node["sub_flow"] = [
                        f"{e.get('from')} -> {e.get('to')} ({e.get('type', 'flow')})"
                        for e in sub_edges
                    ]

        simplified_nodes.append(simplified_node)

    # Extract edges between top-level nodes
    edges_summary = []
    for edge in record.edges:
        edges_summary.append(
            f"{edge.get('source', edge.get('from', ''))} -> {edge.get('target', edge.get('to', ''))} [{edge.get('type', 'flow')}]"
        )

    # Build tree summary
    tree_summary = {
        "example_index": index + 1,
        "query_preview": record.query[:100] if record.query else "",
        "total_nodes": len(record.nodes),
        "total_sections": len(record.sections),
        "nodes": simplified_nodes,
        "top_level_flow": edges_summary,
    }

    # Add error summary for this record
    if record.errors:
        tree_summary["errors"] = [
            {
                "type": e.get("type", "unknown"),
                "description": e.get("description", ""),
                "severity": e.get("severity", "unknown"),
            }
            for e in record.errors[:3]  # Limit to 3 errors per record
        ]

    return tree_summary


def aggregate_analysis_stats(records: List[AnalysisRecord]) -> AggregatedStats:
    """
    Aggregate statistics from multiple analysis records.

    Focuses on:
    - Node types and tree structure patterns
    - Error types and descriptions
    - Basic section statistics

    Args:
        records: List of analysis records

    Returns:
        Aggregated statistics
    """
    stats = AggregatedStats()
    stats.total_analyses = len(records)

    # Temporary lists for calculating averages
    section_counts = []
    overthinking_scores = []

    for idx, record in enumerate(records):
        # Count sections
        section_count = len(record.sections)
        stats.total_sections += section_count
        section_counts.append(section_count)

        # Count node types (including layer2 sub-nodes)
        for node in record.nodes:
            node_type = node.get("type", "unknown")
            stats.node_type_counts[node_type] = (
                stats.node_type_counts.get(node_type, 0) + 1
            )

            # Also count layer2 sub-node types for deeper analysis
            layer2 = node.get("layer2", {})
            if layer2 and "tree" in layer2:
                for sub_node in layer2["tree"].get("nodes", []):
                    sub_type = sub_node.get("type", "unknown")
                    stats.node_type_counts[sub_type] = (
                        stats.node_type_counts.get(sub_type, 0) + 1
                    )

        # Count errors by type (severity tracking is implicit in error details)
        for error in record.errors:
            stats.total_errors += 1
            error_type = error.get("type", "unknown")
            severity = error.get("severity", "unknown")

            stats.error_type_counts[error_type] = (
                stats.error_type_counts.get(error_type, 0) + 1
            )

            # Store example errors with descriptions (limit 5 per type)
            if error_type not in stats.error_examples:
                stats.error_examples[error_type] = []
            if len(stats.error_examples[error_type]) < 5:
                stats.error_examples[error_type].append(
                    {
                        "description": error.get("description", ""),
                        "severity": severity,
                        "details": error.get("details", {}),
                    }
                )

        # Track overthinking scores
        if record.overthinking_analysis:
            score = record.overthinking_analysis.get("overthinking_score", 0.0)
            if isinstance(score, (int, float)):
                overthinking_scores.append(score)
                if score > 0.5:
                    stats.high_overthinking_count += 1

        # Extract simplified reasoning trees (limit to first 5 for prompt)
        if idx < 5:
            tree_summary = extract_simplified_tree(record, idx)
            stats.reasoning_trees.append(tree_summary)

    # Calculate averages
    if overthinking_scores:
        stats.avg_overthinking_score = sum(overthinking_scores) / len(
            overthinking_scores
        )

    if section_counts:
        stats.avg_sections = sum(section_counts) / len(section_counts)

    return stats


# ============================================================================
# CHINESE PROMPTS
# ============================================================================

# Report generation prompt template - Chinese version
REPORT_GENERATION_PROMPT_ZH = """你是一位专门研究大型语言模型(LLM)推理能力的专家研究员。
请根据以下累积的推理特征总结，生成一份关于该推理模型的综合分析报告。

## 分析背景
- **被分析的推理模型**: {reasoning_model}
- **分析使用的模型**: {analysis_model}
- **分析的对话数量**: {total_analyses}
- **总推理节点数**: {total_nodes}

---

## 累积的推理特征总结

以下是通过增量分析多个对话后累积的推理特征总结：

{accumulated_summary}

---

## 节点类型分布

{node_type_distribution}

---

## 错误类型统计

{error_type_distribution}

---

## 分析任务

请基于以上累积的特征总结，生成一份关于该推理模型特点的综合分析报告。报告需包含以下部分：

### 1. 执行摘要
用2-3句话概述该模型的主要推理特点。

### 2. 推理模式分析
基于累积的观察，分析模型的典型推理模式：
- **思维组织方式**：模型如何组织推理步骤？是线性的还是有分支/回溯？
- **验证倾向**：模型是否倾向于反复验证（可能导致overthinking）？
- **问题分解策略**：模型如何将复杂问题分解为子问题？
- **知识调用模式**：模型何时以及如何调用先验知识？

### 3. 错误模式分析
对于观察到的错误模式：
- 该错误类型反映了模型的什么特点？
- 可能的根本原因是什么？
  - 例如：`format_error` 可能说明模型对输出格式的对齐不完全
  - 例如：`overthinking` 可能说明模型倾向于过度验证或缺乏confidence
  - 例如：`logical_error` 可能说明模型的逻辑推理能力有待提升

### 4. 模型特点总结
列出3-5个该推理模型的核心特点，每个特点需要：
- 清晰的特点描述
- 具体的证据支持
- 对用户使用该模型的影响

### 5. 改进建议
基于分析结果，提供3-5条针对性的改进建议：
- 针对训练/微调的建议
- 针对提示工程的建议
- 针对实际使用场景的建议

请用中文撰写报告，使用专业但易懂的语言。确保每个观点都有数据支持。
输出格式为Markdown。
"""

# Incremental summarization prompt - Chinese version
INCREMENTAL_SUMMARY_PROMPT_ZH = """你是一位专门研究大型语言模型(LLM)推理能力的专家研究员。
你的任务是分析推理模型的特征，并维护一个累积的特征总结。

## 分析背景
- **被分析的推理模型**: {reasoning_model}

## 当前累积的特征总结
{previous_summary}

---

## 新的分析数据

以下是新的分析记录，请分析这些记录并更新你的特征总结：

{new_data}

---

## 你的任务

1. **分析新数据**：仔细分析上面的推理树结构和错误信息
2. **识别特征**：识别以下方面的特征：
   - 推理结构模式（节点类型分布、推理深度、分支/回溯模式）
   - 验证行为（是否倾向于反复验证？是否有overthinking倾向？）
   - 错误模式（常见错误类型及其可能原因）
   - 问题分解策略
3. **更新总结**：将新观察到的特征与之前的总结合并

## 输出格式

请输出更新后的累积特征总结，格式如下：

```
### 推理结构特征
- [特征1]: [描述] (出现次数/证据)
- [特征2]: [描述] (出现次数/证据)
...

### 验证行为特征
- [特征1]: [描述] (出现次数/证据)
...

### 错误模式特征
- [错误类型1]: [描述及可能原因] (出现次数)
...

### 问题分解特征
- [特征]: [描述]
...

### 其他观察
- [观察]: [描述]
...
```

注意：
- 合并相似的特征，更新计数
- 保留重要的具体例子作为证据
- 如果新数据与之前的观察矛盾，请注明
- 总结应该简洁但信息丰富
"""

# Memory compression prompt - Chinese version
MEMORY_COMPRESSION_PROMPT_ZH = """你是一位专门研究大型语言模型(LLM)推理能力的专家研究员。

## 任务
当前累积的推理特征总结过长，需要进行压缩。请将以下总结压缩到约 {target_tokens} tokens，同时保留所有关键信息。

## 压缩原则
1. **保留关键特征**：保留所有已识别的推理模式和错误模式的核心描述
2. **合并重复项**：将相似的观察合并，保留计数信息
3. **精简描述**：删除冗余的描述性语言，使用简洁的要点形式
4. **保留证据**：保留最有代表性的例子，删除重复的例子
5. **保持结构**：维持原有的分类结构（推理结构、验证行为、错误模式等）

## 当前总结（需要压缩）

{accumulated_summary}

---

请输出压缩后的总结，保持与原始总结相同的格式结构。
"""


# ============================================================================
# ENGLISH PROMPTS
# ============================================================================

# Report generation prompt template - English version
REPORT_GENERATION_PROMPT_EN = """You are an expert AI researcher specializing in analyzing large language model (LLM) reasoning capabilities.
Based on the accumulated reasoning characteristic summary below, generate a comprehensive analysis report about this reasoning model.

## Analysis Context
- **Reasoning Model Being Analyzed**: {reasoning_model}
- **Analysis Model Used**: {analysis_model}
- **Total Conversations Analyzed**: {total_analyses}
- **Total Reasoning Nodes**: {total_nodes}

---

## Accumulated Reasoning Characteristics Summary

The following is the accumulated summary of reasoning characteristics from incremental analysis of multiple conversations:

{accumulated_summary}

---

## Node Type Distribution

{node_type_distribution}

---

## Error Type Statistics

{error_type_distribution}

---

## Analysis Task

Based on the accumulated characteristic summary above, generate a comprehensive analysis report about this reasoning model. The report should include the following sections:

### 1. Executive Summary
Provide a 2-3 sentence overview of the model's main reasoning characteristics.

### 2. Reasoning Pattern Analysis
Based on accumulated observations, analyze the model's typical reasoning patterns:
- **Thought Organization**: How does the model organize reasoning steps? Linear or with branches/backtracking?
- **Verification Tendency**: Does the model tend to verify repeatedly (potentially leading to overthinking)?
- **Problem Decomposition Strategy**: How does the model break down complex problems into sub-problems?
- **Knowledge Retrieval Patterns**: When and how does the model invoke prior knowledge?

### 3. Error Pattern Analysis
For observed error patterns:
- What characteristics of the model does this error type reflect?
- What are the likely root causes?
  - For example: `format_error` may indicate incomplete alignment with output format requirements
  - For example: `overthinking` may indicate a tendency to over-verify or lack of confidence
  - For example: `logical_error` may indicate room for improvement in logical reasoning ability

### 4. Model Characteristics Summary
List 3-5 core characteristics of this reasoning model, each requiring:
- Clear characteristic description
- Specific supporting evidence
- Impact on users working with this model

### 5. Improvement Recommendations
Based on analysis results, provide 3-5 targeted recommendations:
- Recommendations for training/fine-tuning
- Recommendations for prompt engineering
- Recommendations for practical usage scenarios

Please write the report in English, using professional but accessible language. Ensure every point is supported by data.
Output format: Markdown.
"""

# Incremental summarization prompt - English version
INCREMENTAL_SUMMARY_PROMPT_EN = """You are an expert AI researcher specializing in analyzing large language model (LLM) reasoning capabilities.
Your task is to analyze reasoning model characteristics and maintain an accumulated characteristic summary.

## Analysis Context
- **Reasoning Model Being Analyzed**: {reasoning_model}

## Current Accumulated Characteristic Summary
{previous_summary}

---

## New Analysis Data

The following are new analysis records. Please analyze these records and update your characteristic summary:

{new_data}

---

## Your Task

1. **Analyze New Data**: Carefully analyze the reasoning tree structure and error information above
2. **Identify Characteristics**: Identify characteristics in the following aspects:
   - Reasoning structure patterns (node type distribution, reasoning depth, branching/backtracking patterns)
   - Verification behavior (tendency to verify repeatedly? Overthinking tendencies?)
   - Error patterns (common error types and their possible causes)
   - Problem decomposition strategies
3. **Update Summary**: Merge newly observed characteristics with the previous summary

## Output Format

Please output the updated accumulated characteristic summary in the following format:

```
### Reasoning Structure Characteristics
- [Characteristic 1]: [Description] (occurrence count/evidence)
- [Characteristic 2]: [Description] (occurrence count/evidence)
...

### Verification Behavior Characteristics
- [Characteristic 1]: [Description] (occurrence count/evidence)
...

### Error Pattern Characteristics
- [Error Type 1]: [Description and possible causes] (occurrence count)
...

### Problem Decomposition Characteristics
- [Characteristic]: [Description]
...

### Other Observations
- [Observation]: [Description]
...
```

Notes:
- Merge similar characteristics, update counts
- Retain important specific examples as evidence
- If new data contradicts previous observations, please note this
- The summary should be concise but informative
"""

# Memory compression prompt - English version
MEMORY_COMPRESSION_PROMPT_EN = """You are an expert AI researcher specializing in analyzing large language model (LLM) reasoning capabilities.

## Task
The current accumulated reasoning characteristic summary is too long and needs to be compressed. Please compress the following summary to approximately {target_tokens} tokens while retaining all key information.

## Compression Principles
1. **Retain Key Characteristics**: Preserve core descriptions of all identified reasoning patterns and error patterns
2. **Merge Duplicates**: Combine similar observations, retain count information
3. **Simplify Descriptions**: Remove redundant descriptive language, use concise bullet points
4. **Preserve Evidence**: Keep the most representative examples, remove duplicate examples
5. **Maintain Structure**: Keep the original category structure (reasoning structure, verification behavior, error patterns, etc.)

## Current Summary (to be compressed)

{accumulated_summary}

---

Please output the compressed summary, maintaining the same format structure as the original summary.
"""


# ============================================================================
# PROMPT SELECTION HELPERS
# ============================================================================


def get_report_prompt(language: str = "zh") -> str:
    """Get the report generation prompt for the specified language."""
    if language.lower() in ("en", "english"):
        return REPORT_GENERATION_PROMPT_EN
    return REPORT_GENERATION_PROMPT_ZH


def get_incremental_prompt(language: str = "zh") -> str:
    """Get the incremental summarization prompt for the specified language."""
    if language.lower() in ("en", "english"):
        return INCREMENTAL_SUMMARY_PROMPT_EN
    return INCREMENTAL_SUMMARY_PROMPT_ZH


def get_compression_prompt(language: str = "zh") -> str:
    """Get the memory compression prompt for the specified language."""
    if language.lower() in ("en", "english"):
        return MEMORY_COMPRESSION_PROMPT_EN
    return MEMORY_COMPRESSION_PROMPT_ZH


def get_system_prompt(language: str = "zh", prompt_type: str = "report") -> str:
    """Get the system prompt for the specified language and type."""
    if language.lower() in ("en", "english"):
        if prompt_type == "report":
            return "You are an expert AI researcher specializing in analyzing reasoning model behavior. Generate a comprehensive analysis report based on the accumulated summary."
        elif prompt_type == "compression":
            return "You are an expert AI researcher. Compress the summary while preserving all key information."
        else:
            return "You are an expert AI researcher specializing in LLM reasoning capabilities. Analyze the reasoning model characteristics and update the accumulated summary."
    else:
        if prompt_type == "report":
            return "你是一位专门研究大型语言模型推理能力的专家研究员。请基于累积的特征总结生成综合分析报告。"
        elif prompt_type == "compression":
            return "你是一位专门研究大型语言模型推理能力的专家。请压缩总结，同时保留所有关键信息。"
        else:
            return "你是一位专门研究大型语言模型推理能力的专家。请分析推理模型的特征并更新累积总结。"


def format_record_for_summary(
    record: AnalysisRecord, index: int, language: str = "zh"
) -> str:
    """
    Format a single analysis record for incremental summarization.

    Args:
        record: The analysis record
        index: Index of the record
        language: Language for labels ("zh" or "en")

    Returns:
        Formatted string representation of the record
    """
    is_english = language.lower() in ("en", "english")
    lines = []

    if is_english:
        lines.append(f"### Analysis Record {index + 1}")
        if record.query:
            lines.append(f"**Question**: {record.query[:150]}...")

        # Format nodes (simplified)
        if record.nodes:
            lines.append(f"\n**Reasoning Structure** ({len(record.nodes)} nodes):")
            for node in record.nodes[:5]:
                node_type = node.get("type", "unknown")
                label = node.get("label", "")
                desc = node.get("description", "")[:100]
                lines.append(f"- [{node_type}] {label}: {desc}")

                layer2 = node.get("layer2", {})
                if layer2 and "tree" in layer2:
                    sub_nodes = layer2["tree"].get("nodes", [])[:3]
                    for sn in sub_nodes:
                        lines.append(
                            f"  - [{sn.get('type', '')}] {sn.get('label', '')}"
                        )

        # Format edges (flow)
        if record.edges:
            flow = [
                f"{e.get('source', e.get('from', ''))} -> {e.get('target', e.get('to', ''))}"
                for e in record.edges[:5]
            ]
            lines.append(f"\n**Reasoning Flow**: {' → '.join(flow)}")

        # Format errors
        if record.errors:
            lines.append(f"\n**Detected Errors** ({len(record.errors)}):")
            for err in record.errors[:3]:
                err_type = err.get("type", "unknown")
                severity = err.get("severity", "unknown")
                desc = err.get("description", "")[:150]
                lines.append(f"- [{severity}] {err_type}: {desc}")

                details = err.get("details", {})
                if details.get("redundancy_type"):
                    lines.append(f"  - Redundancy type: {details['redundancy_type']}")
    else:
        lines.append(f"### 分析记录 {index + 1}")
        if record.query:
            lines.append(f"**问题**: {record.query[:150]}...")

        # Format nodes (simplified)
        if record.nodes:
            lines.append(f"\n**推理结构** ({len(record.nodes)} 个节点):")
            for node in record.nodes[:5]:
                node_type = node.get("type", "unknown")
                label = node.get("label", "")
                desc = node.get("description", "")[:100]
                lines.append(f"- [{node_type}] {label}: {desc}")

                layer2 = node.get("layer2", {})
                if layer2 and "tree" in layer2:
                    sub_nodes = layer2["tree"].get("nodes", [])[:3]
                    for sn in sub_nodes:
                        lines.append(
                            f"  - [{sn.get('type', '')}] {sn.get('label', '')}"
                        )

        # Format edges (flow)
        if record.edges:
            flow = [
                f"{e.get('source', e.get('from', ''))} -> {e.get('target', e.get('to', ''))}"
                for e in record.edges[:5]
            ]
            lines.append(f"\n**推理流程**: {' → '.join(flow)}")

        # Format errors
        if record.errors:
            lines.append(f"\n**检测到的错误** ({len(record.errors)} 个):")
            for err in record.errors[:3]:
                err_type = err.get("type", "unknown")
                severity = err.get("severity", "unknown")
                desc = err.get("description", "")[:150]
                lines.append(f"- [{severity}] {err_type}: {desc}")

                details = err.get("details", {})
                if details.get("redundancy_type"):
                    lines.append(f"  - 冗余类型: {details['redundancy_type']}")

    return "\n".join(lines)


def build_incremental_prompt(
    reasoning_model: str,
    previous_summary: str,
    records_batch: List[AnalysisRecord],
    batch_start_index: int,
    language: str = "zh",
) -> str:
    """
    Build prompt for incremental summarization.

    Args:
        reasoning_model: Name of the reasoning model
        previous_summary: Previous accumulated summary
        records_batch: Batch of records to analyze
        batch_start_index: Starting index of this batch
        language: Language for the prompt ("zh" or "en")

    Returns:
        Formatted prompt string
    """
    # Format new data
    new_data_lines = []
    for i, record in enumerate(records_batch):
        new_data_lines.append(
            format_record_for_summary(record, batch_start_index + i, language)
        )
        new_data_lines.append("")  # Empty line between records

    new_data = "\n".join(new_data_lines)

    # Format previous summary
    if not previous_summary:
        if language.lower() in ("en", "english"):
            previous_summary = (
                "(This is the first batch of data, no previous summary available)"
            )
        else:
            previous_summary = "(这是第一批数据，尚无之前的总结)"

    prompt_template = get_incremental_prompt(language)
    return prompt_template.format(
        reasoning_model=reasoning_model,
        previous_summary=previous_summary,
        new_data=new_data,
    )


def build_final_report_prompt(
    reasoning_model: str,
    analysis_model: str,
    accumulated_summary: str,
    stats: AggregatedStats,
    language: str = "zh",
) -> str:
    """
    Build the final report generation prompt.

    Args:
        reasoning_model: Name of the reasoning model being analyzed
        analysis_model: Name of the analysis model used
        accumulated_summary: The accumulated summary from incremental analysis
        stats: Aggregated statistics
        language: Language for the prompt ("zh" or "en")

    Returns:
        Formatted prompt string
    """
    total_nodes = sum(stats.node_type_counts.values())
    is_english = language.lower() in ("en", "english")

    # Format node type distribution
    node_type_lines = []
    for node_type, count in sorted(stats.node_type_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_nodes * 100) if total_nodes > 0 else 0
        if is_english:
            node_type_lines.append(
                f"- **{node_type}**: {count} times ({percentage:.1f}%)"
            )
        else:
            node_type_lines.append(f"- **{node_type}**: {count}次 ({percentage:.1f}%)")

    if is_english:
        node_type_distribution = (
            "\n".join(node_type_lines) if node_type_lines else "No node type data"
        )
    else:
        node_type_distribution = (
            "\n".join(node_type_lines) if node_type_lines else "无节点类型数据"
        )

    # Format error type distribution
    error_type_lines = []
    for error_type, count in sorted(
        stats.error_type_counts.items(), key=lambda x: -x[1]
    ):
        percentage = (count / stats.total_errors * 100) if stats.total_errors > 0 else 0
        if is_english:
            error_type_lines.append(
                f"- **{error_type}**: {count} times ({percentage:.1f}%)"
            )
        else:
            error_type_lines.append(
                f"- **{error_type}**: {count}次 ({percentage:.1f}%)"
            )

    if is_english:
        error_type_distribution = (
            "\n".join(error_type_lines) if error_type_lines else "No errors detected"
        )
    else:
        error_type_distribution = (
            "\n".join(error_type_lines) if error_type_lines else "未检测到错误"
        )

    prompt_template = get_report_prompt(language)
    return prompt_template.format(
        reasoning_model=reasoning_model,
        analysis_model=analysis_model,
        total_analyses=stats.total_analyses,
        total_nodes=total_nodes,
        accumulated_summary=accumulated_summary,
        node_type_distribution=node_type_distribution,
        error_type_distribution=error_type_distribution,
    )


async def generate_analysis_report(
    reasoning_model: str,
    analysis_model: str,
    report_model: str,
    llm_call_func: Callable,
    directories: Optional[List[str]] = None,
    stream: bool = False,
    batch_size: int = 2,  # Number of records to process per batch
    language: str = "zh",  # Language for report generation ("zh" or "en")
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Generate an analysis report for a reasoning model using incremental summarization.

    This uses an agent-like approach similar to error detection:
    1. Process records in batches of 2
    2. Maintain accumulated summary as memory
    3. Generate final report from accumulated summary

    Args:
        reasoning_model: Name of the reasoning model being analyzed
        analysis_model: Name of the analysis model used for analysis
        report_model: Name of the model to use for generating the report
        llm_call_func: Async function to call the LLM
        directories: Optional list of specific directories to include
        stream: Whether to stream the final report response
        batch_size: Number of records to process per batch (default: 2)
        language: Language for report generation ("zh" for Chinese, "en" for English)

    Yields:
        Dict with progress updates and final report
    """
    is_english = language.lower() in ("en", "english")

    # Create session ID for debug logging
    session_id = f"{reasoning_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_id = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)  # Sanitize
    step_counter = 0

    # Step 1: Gather analysis records
    gathering_msg = (
        "Gathering analysis records..." if is_english else "正在收集分析记录..."
    )
    yield {"type": "progress", "stage": "gathering", "message": gathering_msg}

    if directories:
        # Load specific directories
        records = []
        for directory in directories:
            record = load_full_analysis_record(directory)
            if record:
                records.append(record)
    else:
        # Load all records for the specified models
        history = list_analysis_history(
            reasoning_model=reasoning_model, analysis_model=analysis_model, limit=100
        )
        records = []
        for item in history:
            record = load_full_analysis_record(item["directory"])
            if record:
                records.append(record)

    if not records:
        error_msg = (
            "No analysis records found for the specified models"
            if is_english
            else "未找到指定模型的分析记录"
        )
        yield {"type": "error", "message": error_msg}
        return

    if is_english:
        aggregating_msg = (
            f"Found {len(records)} analysis records. Starting incremental analysis..."
        )
    else:
        aggregating_msg = f"找到 {len(records)} 条分析记录。开始增量分析..."

    yield {"type": "progress", "stage": "aggregating", "message": aggregating_msg}

    # Step 2: Aggregate basic statistics (node counts, error counts)
    stats = aggregate_analysis_stats(records)

    # Step 3: Incremental summarization - process records in batches
    accumulated_summary = ""
    total_batches = (len(records) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(records), batch_size):
        batch_end = min(batch_idx + batch_size, len(records))
        batch = records[batch_idx:batch_end]
        batch_num = (batch_idx // batch_size) + 1

        if is_english:
            batch_msg = f"Processing batch {batch_num}/{total_batches} (records {batch_idx + 1}-{batch_end})..."
        else:
            batch_msg = f"处理批次 {batch_num}/{total_batches} (记录 {batch_idx + 1}-{batch_end})..."

        yield {
            "type": "progress",
            "stage": "summarizing",
            "message": batch_msg,
            "batch": batch_num,
            "total_batches": total_batches,
        }

        # Build incremental prompt
        prompt = build_incremental_prompt(
            reasoning_model=reasoning_model,
            previous_summary=accumulated_summary,
            records_batch=batch,
            batch_start_index=batch_idx,
            language=language,
        )

        messages = [
            {"role": "system", "content": get_system_prompt(language, "incremental")},
            {"role": "user", "content": prompt},
        ]

        try:
            # Call LLM for incremental summary (non-streaming for intermediate steps)
            response = await llm_call_func(
                messages=messages,
                model=report_model,
                stream=False,
            )

            # Extract summary from response
            summary_content = ""
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    summary_content = choices[0].get("message", {}).get("content", "")
            else:
                summary_content = str(response)

            # Save debug log for this incremental summary step
            save_llm_call_debug(
                session_id=session_id,
                step_name=f"incremental_summary_batch{batch_num}",
                step_index=step_counter,
                messages=messages,
                response_content=summary_content,
                metadata={
                    "batch_num": batch_num,
                    "total_batches": total_batches,
                    "records_in_batch": len(batch),
                    "batch_start_index": batch_idx,
                    "model": report_model,
                },
            )
            step_counter += 1

            # Update accumulated summary
            accumulated_summary = summary_content

            # Check if memory needs compression during processing
            summary_tokens = estimate_tokens(accumulated_summary)
            log.info(
                "Batch %d/%d completed. Summary length: %d chars (~%d tokens)",
                batch_num,
                total_batches,
                len(accumulated_summary),
                summary_tokens,
            )

            if summary_tokens > MEMORY_MAX_TOKENS:
                if is_english:
                    compress_msg = (
                        f"Summary exceeds {MEMORY_MAX_TOKENS} tokens. Compressing..."
                    )
                else:
                    compress_msg = f"总结超过 {MEMORY_MAX_TOKENS} tokens。正在压缩..."

                yield {
                    "type": "progress",
                    "stage": "compressing",
                    "message": compress_msg,
                }

                compression_prompt = get_compression_prompt(language).format(
                    target_tokens=MEMORY_TARGET_TOKENS,
                    accumulated_summary=accumulated_summary,
                )

                compression_messages = [
                    {
                        "role": "system",
                        "content": get_system_prompt(language, "compression"),
                    },
                    {"role": "user", "content": compression_prompt},
                ]

                try:
                    compression_response = await llm_call_func(
                        messages=compression_messages,
                        model=report_model,
                        stream=False,
                    )

                    compressed_content = ""
                    if isinstance(compression_response, dict):
                        choices = compression_response.get("choices", [])
                        if choices:
                            compressed_content = (
                                choices[0].get("message", {}).get("content", "")
                            )
                    else:
                        compressed_content = str(compression_response)

                    if compressed_content:
                        # Save debug log for compression step
                        save_llm_call_debug(
                            session_id=session_id,
                            step_name=f"compression_during_batch{batch_num}",
                            step_index=step_counter,
                            messages=compression_messages,
                            response_content=compressed_content,
                            metadata={
                                "batch_num": batch_num,
                                "old_tokens": summary_tokens,
                                "new_tokens": estimate_tokens(compressed_content),
                                "model": report_model,
                            },
                        )
                        step_counter += 1

                        old_tokens = summary_tokens
                        accumulated_summary = compressed_content
                        new_tokens = estimate_tokens(accumulated_summary)
                        log.info(
                            "Memory compressed during batch processing: %d -> %d tokens",
                            old_tokens,
                            new_tokens,
                        )
                except Exception as e:
                    log.warning("Failed to compress memory during batch: %s", e)
                    # Continue with uncompressed summary

        except Exception as e:
            log.warning("Error in batch %d: %s", batch_num, e)
            # Continue with what we have
            continue

    # Step 3.5: Check if memory needs compression after all batches
    summary_tokens = estimate_tokens(accumulated_summary)
    if summary_tokens > MEMORY_MAX_TOKENS:
        if is_english:
            compress_msg = f"Summary too long ({summary_tokens} tokens). Compressing to {MEMORY_TARGET_TOKENS} tokens..."
        else:
            compress_msg = f"总结过长 ({summary_tokens} tokens)。正在压缩到 {MEMORY_TARGET_TOKENS} tokens..."

        yield {
            "type": "progress",
            "stage": "compressing",
            "message": compress_msg,
        }

        compression_prompt = get_compression_prompt(language).format(
            target_tokens=MEMORY_TARGET_TOKENS,
            accumulated_summary=accumulated_summary,
        )

        compression_messages = [
            {"role": "system", "content": get_system_prompt(language, "compression")},
            {"role": "user", "content": compression_prompt},
        ]

        try:
            compression_response = await llm_call_func(
                messages=compression_messages,
                model=report_model,
                stream=False,
            )

            compressed_content = ""
            if isinstance(compression_response, dict):
                choices = compression_response.get("choices", [])
                if choices:
                    compressed_content = (
                        choices[0].get("message", {}).get("content", "")
                    )
            else:
                compressed_content = str(compression_response)

            if compressed_content:
                # Save debug log for final compression step
                save_llm_call_debug(
                    session_id=session_id,
                    step_name="compression_final",
                    step_index=step_counter,
                    messages=compression_messages,
                    response_content=compressed_content,
                    metadata={
                        "old_tokens": summary_tokens,
                        "new_tokens": estimate_tokens(compressed_content),
                        "model": report_model,
                    },
                )
                step_counter += 1

                old_tokens = summary_tokens
                accumulated_summary = compressed_content
                new_tokens = estimate_tokens(accumulated_summary)
                log.info("Memory compressed: %d -> %d tokens", old_tokens, new_tokens)
        except Exception as e:
            log.warning("Failed to compress memory: %s", e)
            # Continue with uncompressed summary

    if is_english:
        generating_msg = "Generating final analysis report..."
    else:
        generating_msg = "正在生成最终分析报告..."

    yield {
        "type": "progress",
        "stage": "generating",
        "message": generating_msg,
        "stats": stats.to_dict(),
    }

    # Step 4: Generate final report from accumulated summary
    final_prompt = build_final_report_prompt(
        reasoning_model=reasoning_model,
        analysis_model=analysis_model,
        accumulated_summary=accumulated_summary,
        stats=stats,
        language=language,
    )

    messages = [
        {"role": "system", "content": get_system_prompt(language, "report")},
        {"role": "user", "content": final_prompt},
    ]

    # Log the final prompt size for debugging
    final_prompt_tokens = estimate_tokens(final_prompt)
    log.info(
        "Final report prompt size: %d chars (~%d tokens)",
        len(final_prompt),
        final_prompt_tokens,
    )

    # Save debug log for final report input (before LLM call)
    save_llm_call_debug(
        session_id=session_id,
        step_name="final_report_input",
        step_index=step_counter,
        messages=messages,
        response_content="(pending)",
        metadata={
            "model": report_model,
            "prompt_tokens_estimate": final_prompt_tokens,
            "accumulated_summary_length": len(accumulated_summary),
        },
    )

    try:
        if stream:
            # Stream the final report response
            full_content = ""
            try:
                stream_generator = await llm_call_func(
                    messages=messages,
                    model=report_model,
                    stream=True,
                )
                if stream_generator is None:
                    raise ValueError(
                        "Stream generator is None - LLM call may have failed"
                    )

                async for chunk in stream_generator:
                    if isinstance(chunk, dict):
                        content = chunk.get("content", "")
                    else:
                        content = str(chunk)
                    full_content += content
                    yield {"type": "chunk", "content": content}
            except Exception as stream_error:
                log.error("Streaming error during final report: %s", stream_error)
                # Try to save what we have
                if full_content:
                    log.info("Partial content received: %d chars", len(full_content))
                raise

            # Save debug log for final report generation
            save_llm_call_debug(
                session_id=session_id,
                step_name="final_report",
                step_index=step_counter,
                messages=messages,
                response_content=full_content,
                metadata={
                    "model": report_model,
                    "streaming": True,
                    "records_count": len(records),
                    "reasoning_model": reasoning_model,
                    "analysis_model": analysis_model,
                },
            )
            step_counter += 1

            yield {
                "type": "complete",
                "report": full_content,
                "stats": stats.to_dict(),
                "records_count": len(records),
                "reasoning_model": reasoning_model,
                "analysis_model": analysis_model,
                "accumulated_summary": accumulated_summary,
                "language": language,
            }
        else:
            # Non-streaming response
            response = await llm_call_func(
                messages=messages,
                model=report_model,
                stream=False,
            )

            report_content = ""
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    report_content = choices[0].get("message", {}).get("content", "")
            else:
                report_content = str(response)

            # Save debug log for final report generation
            save_llm_call_debug(
                session_id=session_id,
                step_name="final_report",
                step_index=step_counter,
                messages=messages,
                response_content=report_content,
                metadata={
                    "model": report_model,
                    "streaming": False,
                    "records_count": len(records),
                    "reasoning_model": reasoning_model,
                    "analysis_model": analysis_model,
                },
            )
            step_counter += 1

            yield {
                "type": "complete",
                "report": report_content,
                "stats": stats.to_dict(),
                "records_count": len(records),
                "reasoning_model": reasoning_model,
                "analysis_model": analysis_model,
                "accumulated_summary": accumulated_summary,
                "language": language,
            }

    except Exception as e:
        log.error("Failed to generate report: %s", e)
        error_msg = (
            f"Failed to generate report: {str(e)}"
            if is_english
            else f"生成报告失败: {str(e)}"
        )
        yield {"type": "error", "message": error_msg}
