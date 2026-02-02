"""
Reasoning Analysis Router

This module provides API endpoints for analyzing AI assistant reasoning processes.
It uses a two-layer analysis architecture:
- Layer 1: Coarse-grained analysis (problem decomposition, solution paths, verification)
- Layer 2: Fine-grained analysis (atomic steps with action type classification)
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.responses import StreamingResponse

from open_webui.constants import ERROR_MESSAGES
from open_webui.models.chats import Chats
from open_webui.utils.auth import get_verified_user
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.models import get_all_models
from open_webui.utils.misc import (
    get_content_from_message,
    get_last_user_message_item,
    get_message_list,
)
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from .reasoning_analysis import (
    ReasoningAnalysisRequest,
    analyze_layer1,
    build_final_result,
    extract_reasoning_and_answer,
)
from .reasoning_analysis.core import analyze_layer2_for_node
from .reasoning_analysis.error_detection.agent import (
    AgentConfig,
    SectionAnalysisAgent,
    AgentDetectedError,
    AgentErrorType,
    AgentErrorSeverity,
)
from .reasoning_analysis.error_detection.solutions import (
    get_solutions_for_error_type,
    get_all_error_types,
    get_training_methods,
    add_training_method,
    get_knowledge_base,
)
from .reasoning_analysis.utils import (
    persist_analysis_artifact,
    get_cached_analysis,
    get_message_analysis_dir,
)
from .reasoning_analysis.extractors import (
    format_sections_for_prompt,
    extract_completion_content,
    get_section_range_text,
)
from .reasoning_analysis.report_generator import (
    list_analysis_history,
    load_full_analysis_record,
    aggregate_analysis_stats,
    generate_analysis_report,
    list_saved_reports,
    get_saved_report,
    delete_saved_report,
)

log = logging.getLogger(__name__)
router = APIRouter()

# Configuration
STAGE_TIMEOUT_SECONDS = 600
LOG_ANALYSIS_RESULTS = os.getenv("REASONING_ANALYSIS_LOGS", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ANALYSIS_LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "analysis_logs"


@router.post("/reasoning/stream")
async def analyze_reasoning_trace_stream(
    form_data: ReasoningAnalysisRequest,
    request: Request,
    user=Depends(get_verified_user),
):
    """
    Analyze a reasoning trace using two-layer analysis with streaming response.

    This endpoint streams results progressively:
    1. First sends Layer 1 (coarse-grained) analysis when complete
    2. Then sends Layer 2 (fine-grained) analysis for each node as they complete
    3. Finally sends the complete merged result

    Each event is sent as Server-Sent Events (SSE) with the following format:
    - event: layer1 | layer2_node | complete | error
    - data: JSON payload
    """
    log.info(
        "Streaming reasoning analysis requested | chat=%s message=%s model=%s",
        form_data.chat_id,
        form_data.message_id,
        form_data.model,
    )

    # Validate chat access (same as non-streaming endpoint)
    chat = Chats.get_chat_by_id(form_data.chat_id)
    if chat is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.DEFAULT()
        )

    if user.role == "user" and chat.user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )

    # Extract message content
    messages_map = chat.chat.get("history", {}).get("messages", {}) or {}
    message = messages_map.get(form_data.message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=ERROR_MESSAGES.DEFAULT()
        )

    message_chain = get_message_list(messages_map, form_data.message_id)
    last_user_message = get_last_user_message_item(message_chain) or {}
    question_text = get_content_from_message(last_user_message) or ""

    raw_content = get_content_from_message(message) or ""
    reasoning_text, final_answer = extract_reasoning_and_answer(raw_content)
    pre_computed_sections = message.get("reasoning_sections")

    # Get the reasoning model (the model that generated this message)
    reasoning_model = message.get("model", "unknown")

    metadata_base = {
        "analysis_source": "reasoning_trace_stream",
        "chat_id": form_data.chat_id,
        "message_id": form_data.message_id,
    }
    stage_metadata = {
        **metadata_base,
        "model": form_data.model,
        "reasoning_model": reasoning_model,
    }

    # Check cache first (unless force flag is set)
    force_analysis = getattr(form_data, "force", False)
    if not force_analysis:
        cached_result = get_cached_analysis(
            ANALYSIS_LOG_DIR,
            form_data.message_id,
            form_data.model,
        )
        if cached_result:
            log.info(
                "Returning cached analysis result (streaming) | chat=%s message=%s model=%s",
                form_data.chat_id,
                form_data.message_id,
                form_data.model,
            )
            # Return cached result as a single 'cached' event followed by 'complete'
            cached_payload = cached_result.get("payload", cached_result)

            # Also try to load cached error detection results
            cached_error_detection = None
            try:
                message_dir = get_message_analysis_dir(
                    ANALYSIS_LOG_DIR,
                    form_data.message_id,
                    form_data.model,
                )
                error_detection_file = message_dir / "04_error_detection_result.json"
                if error_detection_file.exists():
                    with error_detection_file.open("r", encoding="utf-8") as f:
                        error_data = json.load(f)
                        cached_error_detection = error_data.get("payload", error_data)
                        log.info(
                            "Loaded cached error detection results from %s",
                            error_detection_file,
                        )
            except Exception as e:
                log.warning("Failed to load cached error detection results: %s", e)

            async def cached_generator() -> AsyncGenerator[str, None]:
                yield f"event: cached\ndata: {json.dumps({'status': 'cached'})}\n\n"
                yield f"event: complete\ndata: {json.dumps(cached_payload)}\n\n"

                # Send cached error detection results if available
                if cached_error_detection:
                    errors_list = cached_error_detection.get("errors", [])
                    overthinking_score = cached_error_detection.get(
                        "overthinking_score", 0.0
                    )
                    first_correct_section = cached_error_detection.get(
                        "first_correct_answer_section"
                    )
                    verification_sections = cached_error_detection.get(
                        "verification_sections", []
                    )
                    all_answer_sections = cached_error_detection.get(
                        "all_answer_sections", []
                    )
                    total_sections_analyzed = cached_error_detection.get(
                        "total_sections", 0
                    )
                    accumulated_summary = cached_error_detection.get(
                        "accumulated_summary", ""
                    )

                    detection_payload = {
                        "errors": errors_list,
                        "overthinking_analysis": {
                            "score": overthinking_score,
                            "first_correct_answer_section": first_correct_section,
                            "verification_sections": verification_sections,
                            "all_answer_sections": all_answer_sections,
                            "total_sections": total_sections_analyzed,
                        },
                        "accumulated_summary": accumulated_summary,
                    }
                    yield f"event: error_detection\ndata: {json.dumps(detection_payload)}\n\n"

            return StreamingResponse(
                cached_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

    async def stream_generator() -> AsyncGenerator[str, None]:
        """Generator for streaming analysis results."""
        try:
            # Ensure models are loaded
            if not request.app.state.MODELS:
                try:
                    await get_all_models(request, user=user)
                except Exception as exc:
                    log.exception("Unable to load models before streaming analysis")
                    yield f"event: error\ndata: {json.dumps({'error': 'Unable to load models'})}\n\n"
                    return

            # Helper function to run analysis stages with streaming LLM call
            # This collects the streaming response but avoids the timeout issue
            # by keeping the connection active during generation
            async def _run_stage_streaming(stage: str, messages: list[dict]) -> dict:
                """
                Run an analysis stage with streaming LLM call.

                Uses stream=True internally to keep the connection alive and avoid
                timeout issues, but collects the complete response before returning.
                """
                from starlette.responses import StreamingResponse

                payload = {
                    "model": form_data.model,
                    "messages": messages,
                    "stream": True,  # Use streaming to avoid timeout
                    "temperature": 0,
                    "metadata": {**metadata_base, "analysis_stage": stage},
                }

                log.info("Starting streaming stage '%s'", stage)
                # Use bypass_filter=True to allow all verified users to use analysis models
                # The user has already been verified via get_verified_user dependency
                response = await generate_chat_completion(
                    request, payload, user, bypass_filter=True
                )

                if isinstance(response, StreamingResponse):
                    # Collect streaming response into complete content
                    accumulated_content = ""
                    try:
                        async for chunk in response.body_iterator:
                            if isinstance(chunk, bytes):
                                chunk = chunk.decode("utf-8")
                            if chunk.startswith("data: "):
                                try:
                                    data_str = chunk[6:].strip()
                                    if data_str and data_str != "[DONE]":
                                        data = json.loads(data_str)
                                        if (
                                            "choices" in data
                                            and len(data["choices"]) > 0
                                        ):
                                            choice = data["choices"][0]
                                            delta = choice.get("delta", {})
                                            chunk_content = delta.get("content", "")
                                            if chunk_content:
                                                accumulated_content += chunk_content
                                            if choice.get("finish_reason"):
                                                break
                                except json.JSONDecodeError:
                                    pass
                    except Exception as e:
                        log.warning(
                            "Error collecting streaming response for stage '%s': %s",
                            stage,
                            e,
                        )

                    # Return in standard OpenAI response format
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": accumulated_content,
                                },
                                "finish_reason": "stop",
                            }
                        ]
                    }
                else:
                    # Non-streaming response (shouldn't happen but handle gracefully)
                    return response

            # Wrapper for stage execution with timeout and error handling
            async def _run_stage_safe(stage: str, messages: list[dict]) -> dict:
                started = time.monotonic()
                try:
                    result = await asyncio.wait_for(
                        _run_stage_streaming(stage, messages),
                        timeout=STAGE_TIMEOUT_SECONDS,
                    )
                    log.info(
                        "Stage '%s' completed in %.2fs",
                        stage,
                        time.monotonic() - started,
                    )
                    return result
                except asyncio.TimeoutError:
                    raise HTTPException(
                        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                        detail=f"Stage '{stage}' timed out",
                    )
                except Exception as exc:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"Stage '{stage}' failed: {exc}",
                    )

            # Send start event with stages info
            yield f"event: start\ndata: {json.dumps({'status': 'analyzing', 'stage': 'layer1', 'stages': ['layer1', 'layer2', 'error_detection']})}\n\n"

            # ================================================================
            # Save Input Data for debugging
            # ================================================================
            reasoning_with_sections = (
                format_sections_for_prompt(pre_computed_sections)
                if pre_computed_sections
                else ""
            )
            input_data = {
                "question": question_text,
                "final_answer": final_answer,
                "reasoning_model": reasoning_model,  # The model being analyzed
                "reasoning_text_length": len(reasoning_text),
                "reasoning_text": (
                    reasoning_text[:5000] + "..."
                    if len(reasoning_text) > 5000
                    else reasoning_text
                ),
                "pre_computed_sections_count": (
                    len(pre_computed_sections) if pre_computed_sections else 0
                ),
                "pre_computed_sections": pre_computed_sections,
                "formatted_reasoning_with_sections": (
                    reasoning_with_sections[:10000] + "..."
                    if len(reasoning_with_sections) > 10000
                    else reasoning_with_sections
                ),
            }
            persist_analysis_artifact(
                "00_input",
                input_data,
                stage_metadata,
                ANALYSIS_LOG_DIR,
                enabled=True,  # Always save for debugging
            )

            # ================================================================
            # LAYER 1 ANALYSIS (Start first, render immediately when done)
            # ================================================================
            log.info("Streaming: Starting Layer 1 analysis")

            # Start Layer 1 analysis
            try:
                layer1_result = await analyze_layer1(
                    question=question_text,
                    reasoning_text=reasoning_text,
                    final_answer=final_answer,
                    run_stage_func=_run_stage_safe,
                    prompt_override=form_data.prompt_override,
                    sections=pre_computed_sections,
                )
                layer1_json, layer1_response, sections = layer1_result
            except Exception as e:
                log.exception("Layer 1 analysis failed with exception")
                yield f"event: error\ndata: {json.dumps({'error': f'Layer 1 failed: {str(e)}'})}\n\n"
                return

            # Process Layer 1 result immediately
            if not isinstance(layer1_json, dict):
                # Save the raw response for debugging when JSON parsing fails
                raw_content = extract_completion_content(layer1_response)
                persist_analysis_artifact(
                    "layer1_parse_error",
                    {
                        "error": "Failed to parse JSON from model output",
                        "raw_response": layer1_response,
                        "extracted_content": raw_content,
                        "content_length": len(raw_content) if raw_content else 0,
                    },
                    stage_metadata,
                    ANALYSIS_LOG_DIR,
                    enabled=True,
                )
                yield f"event: error\ndata: {json.dumps({'error': 'Layer 1 analysis failed to produce valid JSON'})}\n\n"
                return

            layer1_nodes = layer1_json.get("nodes", [])
            layer1_edges = layer1_json.get("edges", [])

            # Enrich layer1 nodes with section text
            from .reasoning_analysis.extractors import get_section_range_text

            enriched_layer1_nodes = []
            for node in layer1_nodes:
                enriched_node = node.copy()
                section_start = node.get("section_start")
                section_end = node.get("section_end")
                if section_start and section_end:
                    segment_text, _, _ = get_section_range_text(
                        sections, section_start, section_end
                    )
                    if segment_text:
                        enriched_node["segment_text"] = segment_text
                enriched_layer1_nodes.append(enriched_node)

            layer1_payload = {
                "nodes": enriched_layer1_nodes,
                "edges": layer1_edges,
                "sections": sections,
                "root_summary": layer1_json.get("root_summary", ""),
                "answer_summary": layer1_json.get("answer_summary", ""),
            }

            # Save Layer 1 result for debugging
            persist_analysis_artifact(
                "01_layer1_result",
                {
                    "raw_response": layer1_json,
                    "enriched_payload": layer1_payload,
                    "nodes_count": len(layer1_nodes),
                    "edges_count": len(layer1_edges),
                    "sections_count": len(sections) if sections else 0,
                },
                stage_metadata,
                ANALYSIS_LOG_DIR,
                enabled=True,
            )

            # *** IMMEDIATELY YIELD Layer 1 results ***
            log.info(
                "Streaming: Layer 1 completed with %d nodes, %d edges - sending to frontend immediately",
                len(layer1_nodes),
                len(layer1_edges),
            )
            yield f"event: layer1\ndata: {json.dumps(layer1_payload)}\n\n"

            # ================================================================
            # Start Layer 2 and Error Detection
            # Layer 2 runs first, then Error Detection with streaming progress
            # ================================================================

            # Prepare LLM call function for error detection
            async def llm_call_func(model, messages, tools=None):
                return await _run_stage_safe("error_detection", messages)

            # Prepare sections for error detection
            all_sections_for_detection = []
            for i, section in enumerate(sections, start=1):
                all_sections_for_detection.append(
                    {
                        "section_number": i,
                        "text": section.get("text", section.get("content", "")),
                    }
                )

            # Error detection result holder
            error_detection_result = {
                "errors": [],
                "overthinking_score": 0.0,
                "first_correct_answer_section": None,
                "verification_sections": [],
                "all_answer_sections": [],
                "total_sections": len(sections),
                "accumulated_summary": "",
                "batch_results": [],
                "claims": [],
                "query_answers": [],
            }

            # Run error detection with streaming (to avoid timeout)
            async def run_error_detection_with_streaming():
                """Run error detection using streaming LLM calls and collect results."""
                nonlocal error_detection_result

                try:
                    # Get the analysis directory for this message to save history
                    message_analysis_dir = get_message_analysis_dir(
                        ANALYSIS_LOG_DIR,
                        form_data.message_id,
                        form_data.model,
                    )

                    main_agent = SectionAnalysisAgent(
                        config=AgentConfig(
                            batch_size=10,
                            model=form_data.model,
                            history_output_dir=str(message_analysis_dir),
                        )
                    )

                    # Use streaming version - collect events
                    async for event in main_agent.analyze_all_sections_streaming(
                        sections=all_sections_for_detection,
                        query=question_text,
                        expected_answer=final_answer,
                        llm_call_func=llm_call_func,
                    ):
                        event_type = event.get("type", "")

                        if event_type == "complete":
                            result = event.get("result", {})

                            # Get overthinking analysis
                            overthinking_analysis = result.get(
                                "overthinking_analysis", {}
                            )
                            overthinking_score = overthinking_analysis.get(
                                "overthinking_score", 0.0
                            )
                            first_correct_section = overthinking_analysis.get(
                                "initial_answer_section"
                            )

                            # Build error objects
                            errors = []
                            for error_dict in result.get("errors", []):
                                try:
                                    error = AgentDetectedError(
                                        type=AgentErrorType(
                                            error_dict.get("type", "Logical Error")
                                        ),
                                        description=error_dict.get("description", ""),
                                        severity=AgentErrorSeverity(
                                            error_dict.get("severity", "medium")
                                        ),
                                        section_numbers=error_dict.get(
                                            "section_numbers", []
                                        ),
                                        context=error_dict.get("context"),
                                        details=error_dict.get("details", {}),
                                    )

                                    # Map to layer1 nodes
                                    error_sections = error.section_numbers
                                    if error_sections:
                                        for node in layer1_nodes:
                                            node_start = node.get("section_start")
                                            node_end = node.get("section_end")
                                            if node_start and node_end:
                                                if any(
                                                    node_start <= sec <= node_end
                                                    for sec in error_sections
                                                ):
                                                    error.details = error.details or {}
                                                    error.details["node_id"] = node.get(
                                                        "id"
                                                    )
                                                    error.details["node_type"] = (
                                                        node.get("type")
                                                    )
                                                    error.details["node_label"] = (
                                                        node.get("label")
                                                    )
                                                    break

                                    errors.append(error)
                                except Exception as e:
                                    log.warning("Failed to parse error: %s", e)

                            # Update result holder
                            error_detection_result.update(
                                {
                                    "errors": errors,
                                    "overthinking_score": overthinking_score,
                                    "first_correct_answer_section": first_correct_section,
                                    "verification_sections": overthinking_analysis.get(
                                        "verification_sections", []
                                    ),
                                    "all_answer_sections": overthinking_analysis.get(
                                        "all_answer_sections", []
                                    ),
                                    "total_sections": len(sections),
                                    "accumulated_summary": result.get(
                                        "accumulated_summary", ""
                                    ),
                                    "batch_results": result.get("batch_results", []),
                                    "claims": result.get("claims", []),
                                    "query_answers": result.get("query_answers", []),
                                }
                            )

                            log.info(
                                "Streaming error detection complete: %d errors, overthinking_score=%.2f",
                                len(errors),
                                overthinking_score,
                            )

                except Exception as e:
                    log.warning("Streaming error detection failed: %s", e)

            # Create error detection task to run in background
            error_detection_task = asyncio.create_task(
                run_error_detection_with_streaming()
            )

            # Send status update for Layer 2
            yield f"event: status\ndata: {json.dumps({'status': 'analyzing', 'stage': 'layer2', 'total_nodes': len(layer1_nodes)})}\n\n"

            # ================================================================
            # LAYER 2: Fine-Grained Analysis (Progressive)
            # ================================================================
            log.info(
                "Streaming: Starting Layer 2 analysis for %d nodes", len(layer1_nodes)
            )

            expandable_types = [
                "problem_decomposition",
                "reasoning_step",
                "verification",
                "problem_analysis",
                "solution_path",
                "check",
            ]
            expandable_nodes = [
                node for node in layer1_nodes if node.get("type") in expandable_types
            ]

            layer2_data = {}
            completed_count = 0

            # Create tasks for parallel processing
            async def process_node(idx, node):
                """Process a single node and return result tuple."""
                node_id = node.get("id", "")
                try:
                    # Analyze this node
                    node_result = await analyze_layer2_for_node(
                        node=node,
                        question=question_text,
                        sections=sections,
                        run_stage_func=_run_stage_safe,
                        reasoning_text=reasoning_text,
                        final_answer=final_answer,
                    )
                    return (idx, node, node_result, None)
                except Exception as node_exc:
                    log.warning(
                        "Streaming: Layer 2 failed for node %s: %s", node_id, node_exc
                    )
                    return (idx, node, None, str(node_exc))

            # Process all nodes in parallel
            tasks = [
                process_node(idx, node) for idx, node in enumerate(expandable_nodes)
            ]
            results = await asyncio.gather(*tasks)

            # Stream results as they complete (sorted by index for consistent ordering)
            for idx, node, node_result, error in results:
                node_id = node.get("id", "")

                if node_result:
                    layer2_data[node_id] = node_result

                    # Save each Layer 2 node result for debugging
                    persist_analysis_artifact(
                        f"02_layer2_node_{idx:02d}_{node_id}",
                        {
                            "node_id": node_id,
                            "node_type": node.get("type", ""),
                            "node_label": node.get("label", ""),
                            "section_start": node.get("section_start"),
                            "section_end": node.get("section_end"),
                            "layer2_result": node_result,
                            "steps_count": len(node_result.get("steps", [])),
                            "issues_count": len(node_result.get("issues", [])),
                        },
                        stage_metadata,
                        ANALYSIS_LOG_DIR,
                        enabled=True,  # Always save for debugging
                    )

                    # Stream this node's result
                    node_payload = {
                        "node_id": node_id,
                        "layer2": node_result,
                        "completed": completed_count + 1,
                        "total": len(expandable_nodes),
                    }
                    yield f"event: layer2_node\ndata: {json.dumps(node_payload)}\n\n"
                elif error:
                    # Stream error event
                    yield f"event: layer2_error\ndata: {json.dumps({'node_id': node_id, 'error': error})}\n\n"

                completed_count += 1

            log.info(
                "Streaming: Layer 2 completed for %d/%d nodes",
                len(layer2_data),
                len(expandable_nodes),
            )

            # ================================================================
            # Build and Send Final Merged Result
            # ================================================================
            merged_payload = build_final_result(layer1_json, layer2_data, sections)

            # Save final merged result with clear naming
            persist_analysis_artifact(
                "03_final_merged_result",
                {
                    "merged_payload": merged_payload,
                    "layer1_nodes_count": len(layer1_nodes),
                    "layer2_nodes_processed": len(layer2_data),
                    "total_layer2_steps": merged_payload.get(
                        "analysis_metadata", {}
                    ).get("total_layer2_steps", 0),
                    "total_issues": merged_payload.get("analysis_metadata", {}).get(
                        "total_issues", 0
                    ),
                },
                stage_metadata,
                ANALYSIS_LOG_DIR,
                enabled=True,  # Always enable for debugging
            )

            # Also save as "final_merged_result" for cache compatibility
            persist_analysis_artifact(
                "final_merged_result",  # Use same name as non-streaming for cache compatibility
                merged_payload,
                stage_metadata,
                ANALYSIS_LOG_DIR,
                enabled=True,  # Always enable for caching
            )

            # ================================================================
            # Wait for Error Detection to Complete
            # ================================================================
            # Initialize variables before try block to avoid NameError if exception occurs
            agentic_errors = []
            overthinking_score = 0.0

            try:
                await error_detection_task

                # Get results from the error_detection_result holder (updated by the streaming task)
                agentic_errors = error_detection_result.get("errors", [])
                overthinking_score = error_detection_result.get(
                    "overthinking_score", 0.0
                )
                first_correct_section = error_detection_result.get(
                    "first_correct_answer_section"
                )
                verification_sections = error_detection_result.get(
                    "verification_sections", []
                )
                all_answer_sections = error_detection_result.get(
                    "all_answer_sections", []
                )
                total_sections_analyzed = error_detection_result.get(
                    "total_sections", 0
                )
                accumulated_summary = error_detection_result.get(
                    "accumulated_summary", ""
                )
                claims = error_detection_result.get("claims", [])
                query_answers = error_detection_result.get("query_answers", [])

                log.info(
                    "Streaming: Error detection completed with %d errors, overthinking_score=%.2f, verifications=%s, claims=%d, query_answers=%d",
                    len(agentic_errors),
                    overthinking_score,
                    verification_sections,
                    len(claims),
                    len(query_answers),
                )

                # Convert AgentDetectedError objects to dicts for JSON serialization
                errors_dict_list = [error.to_dict() for error in agentic_errors]

                # Save error detection results
                persist_analysis_artifact(
                    "04_error_detection_result",
                    {
                        "errors": errors_dict_list,
                        "error_count": len(agentic_errors),
                        "overthinking_score": overthinking_score,
                        "first_correct_answer_section": first_correct_section,
                        "verification_sections": verification_sections,
                        "all_answer_sections": all_answer_sections,
                        "total_sections": total_sections_analyzed,
                        "accumulated_summary": accumulated_summary,
                    },
                    stage_metadata,
                    ANALYSIS_LOG_DIR,
                    enabled=True,
                )

                # Send error detection results to frontend with enhanced overthinking analysis
                detection_payload = {
                    "errors": errors_dict_list,
                    "overthinking_analysis": {
                        "score": overthinking_score,
                        "first_correct_answer_section": first_correct_section,
                        "verification_sections": verification_sections,
                        "all_answer_sections": all_answer_sections,
                        "total_sections": total_sections_analyzed,
                    },
                    "accumulated_summary": accumulated_summary,
                    "claims": claims,
                    "query_answers": query_answers,
                }
                yield f"event: error_detection\ndata: {json.dumps(detection_payload)}\n\n"

            except Exception as e:
                log.warning("Error detection task failed: %s", str(e))
                error_payload = {
                    "errors": [],
                    "overthinking_analysis": {
                        "score": 0.0,
                        "first_correct_answer_section": None,
                        "verification_sections": [],
                        "all_answer_sections": [],
                        "total_sections": 0,
                    },
                    "accumulated_summary": "",
                    "warning": str(e),
                }
                yield f"event: error_detection\ndata: {json.dumps(error_payload)}\n\n"

            log.info(
                "Streaming analysis completed | layer1_nodes=%d layer2_nodes=%d errors=%d overthinking_score=%.2f | chat=%s message=%s",
                len(layer1_nodes),
                len(layer2_data),
                len(agentic_errors),
                overthinking_score,
                form_data.chat_id,
                form_data.message_id,
            )

            yield f"event: complete\ndata: {json.dumps(merged_payload)}\n\n"

        except HTTPException as he:
            yield f"event: error\ndata: {json.dumps({'error': he.detail})}\n\n"
        except Exception as exc:
            log.exception("Streaming analysis failed unexpectedly")
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/reasoning/cache/{message_id}")
async def clear_analysis_cache(
    message_id: str,
    user=Depends(get_verified_user),
):
    """
    Clear cached analysis results for a specific message.

    This is useful when the user wants to re-analyze with a different model
    or force a fresh analysis.
    """
    try:
        import shutil

        # Find all directories for this message (across different models)
        deleted_count = 0
        if ANALYSIS_LOG_DIR.exists():
            safe_message = re.sub(r"[^a-zA-Z0-9_-]+", "_", message_id)
            for item in ANALYSIS_LOG_DIR.iterdir():
                if item.is_dir() and item.name.startswith(f"{safe_message}_"):
                    shutil.rmtree(item)
                    deleted_count += 1
                    log.info("Deleted analysis cache directory: %s", item)

        return {
            "success": True,
            "message": f"Cleared {deleted_count} cached analysis result(s)",
            "deleted_count": deleted_count,
        }
    except Exception as exc:
        log.exception("Failed to clear analysis cache for message=%s", message_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {exc}",
        )


# =============================================================================
# Error Solutions Knowledge Base API
# =============================================================================


class ErrorSolutionResponse(BaseModel):
    """Response model for error solutions."""

    error_type: str
    found: bool
    display_name: Optional[str] = None
    description: Optional[str] = None
    severity_default: Optional[str] = None
    quick_fixes: List[str] = []
    # Test-time scaling methods (no training required)
    test_time_methods: List[Dict[str, Any]] = []
    test_time_methods_by_category: Dict[str, List[Dict[str, Any]]] = {}
    test_time_categories: List[str] = []
    # Training-required methods
    training_methods: List[Dict[str, Any]] = []
    training_methods_by_category: Dict[str, List[Dict[str, Any]]] = {}
    categories: List[str] = []
    evaluation_metrics: List[str] = []


class AddTrainingMethodRequest(BaseModel):
    """Request model for adding a training method."""

    error_type: str
    method: Dict[str, Any] = Field(
        ...,
        description="Training method with fields: name, description, category, effect, reference, difficulty",
    )


class AddQuickFixRequest(BaseModel):
    """Request model for adding a quick fix."""

    error_type: str
    quick_fix: str


@router.get("/reasoning/solutions/error-types")
async def get_error_types_list(
    user=Depends(get_verified_user),
) -> Dict[str, Any]:
    """
    Get list of all error types in the knowledge base.

    Returns:
        Dictionary with list of error types and their basic info
    """
    try:
        error_types = get_all_error_types()
        kb = get_knowledge_base()

        types_info = []
        for et in error_types:
            info = kb.get_error_type_info(et)
            if info:
                types_info.append(
                    {
                        "name": et,
                        "display_name": info.get("display_name", et),
                        "description": info.get("description", ""),
                        "severity_default": info.get("severity_default", "medium"),
                        "quick_fixes_count": len(info.get("quick_fixes", [])),
                        "training_methods_count": len(info.get("training_methods", [])),
                    }
                )

        return {
            "success": True,
            "error_types": types_info,
            "total": len(types_info),
        }
    except Exception as e:
        log.error("Failed to get error types: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/reasoning/solutions/{error_type}")
async def get_error_solutions(
    error_type: str,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    user=Depends(get_verified_user),
) -> ErrorSolutionResponse:
    """
    Get solutions for a specific error type.

    Args:
        error_type: The error type to get solutions for
        category: Optional filter by category (e.g., "RL-based Length Reward")
        difficulty: Optional filter by difficulty (beginner, intermediate, advanced)

    Returns:
        Complete solutions information for the error type
    """
    try:
        solutions = get_solutions_for_error_type(error_type)

        # Apply filters if specified
        if solutions["found"] and (category or difficulty):
            methods = get_training_methods(error_type, category, difficulty)
            solutions["training_methods"] = methods

            # Rebuild categories dict with filtered methods
            categories = {}
            for method in methods:
                cat = method.get("category", "Other")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(method)
            solutions["training_methods_by_category"] = categories
            solutions["categories"] = list(categories.keys())

        return ErrorSolutionResponse(**solutions)
    except Exception as e:
        log.error("Failed to get solutions for %s: %s", error_type, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/reasoning/solutions/{error_type}/training-methods")
async def get_error_training_methods(
    error_type: str,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    user=Depends(get_verified_user),
) -> Dict[str, Any]:
    """
    Get training methods for a specific error type.

    Args:
        error_type: The error type to get methods for
        category: Optional filter by category
        difficulty: Optional filter by difficulty level

    Returns:
        List of training methods with metadata
    """
    try:
        methods = get_training_methods(error_type, category, difficulty)
        kb = get_knowledge_base()
        categories = kb.get_training_method_categories(error_type)

        return {
            "success": True,
            "error_type": error_type,
            "methods": methods,
            "total": len(methods),
            "available_categories": categories,
            "difficulty_levels": kb.get_difficulty_levels(),
        }
    except Exception as e:
        log.error("Failed to get training methods: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/reasoning/solutions/add-method")
async def add_error_training_method(
    request: AddTrainingMethodRequest,
    user=Depends(get_verified_user),
) -> Dict[str, Any]:
    """
    Add a new training method to an error type.

    The method should include:
    - name: Short name for the method (required)
    - description: Description of what it does (required)
    - category: Category grouping (e.g., "RL-based Length Reward")
    - effect: What effect/improvement it provides
    - reference: Paper or implementation reference
    - difficulty: beginner, intermediate, or advanced
    - full_name: Full name if abbreviated
    """
    try:
        success = add_training_method(request.error_type, request.method)

        if success:
            return {
                "success": True,
                "message": f"Added method '{request.method.get('name')}' to {request.error_type}",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to add method. Ensure error type '{request.error_type}' exists and method has required fields.",
            )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to add training method: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/reasoning/solutions/add-quick-fix")
async def add_error_quick_fix(
    request: AddQuickFixRequest,
    user=Depends(get_verified_user),
) -> Dict[str, Any]:
    """
    Add a new quick fix suggestion to an error type.
    """
    try:
        kb = get_knowledge_base()
        success = kb.add_quick_fix(request.error_type, request.quick_fix)

        if success:
            return {
                "success": True,
                "message": f"Added quick fix to {request.error_type}",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to add quick fix. Ensure error type '{request.error_type}' exists.",
            )
    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to add quick fix: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/reasoning/solutions/export/all")
async def export_knowledge_base(
    user=Depends(get_verified_user),
) -> Dict[str, Any]:
    """
    Export the entire knowledge base.

    Useful for backup or transfer to another instance.
    """
    try:
        kb = get_knowledge_base()
        data = kb.export_data()

        return {
            "success": True,
            "data": data,
            "version": data.get("version", "unknown"),
            "last_updated": data.get("last_updated", "unknown"),
        }
    except Exception as e:
        log.error("Failed to export knowledge base: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/reasoning/solutions/reload")
async def reload_knowledge_base(
    user=Depends(get_verified_user),
) -> Dict[str, Any]:
    """
    Reload the knowledge base from disk.

    Useful after manually editing the JSON file.
    """
    try:
        kb = get_knowledge_base()
        kb.reload()

        return {
            "success": True,
            "message": "Knowledge base reloaded",
            "error_types": kb.get_all_error_types(),
        }
    except Exception as e:
        log.error("Failed to reload knowledge base: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# =============================================================================
# Model Analysis Report API
# =============================================================================


class AnalysisHistoryRequest(BaseModel):
    """Request model for listing analysis history."""

    reasoning_model: Optional[str] = Field(
        None, description="Filter by reasoning model name"
    )
    analysis_model: Optional[str] = Field(
        None, description="Filter by analysis model name"
    )
    limit: int = Field(
        100, ge=1, le=500, description="Maximum number of records to return"
    )


class ReportGenerationRequest(BaseModel):
    """Request model for generating analysis report."""

    reasoning_model: str = Field(
        ..., description="Name of the reasoning model being analyzed"
    )
    analysis_model: Optional[str] = Field(
        None, description="Name of the analysis model used"
    )
    report_model: str = Field(..., description="Model to use for generating the report")
    directories: Optional[List[str]] = Field(
        None, description="Specific analysis directories to include"
    )
    stream: bool = Field(True, description="Whether to stream the response")
    language: str = Field(
        "zh", description="Language for the report ('zh' for Chinese, 'en' for English)"
    )


@router.get("/reasoning/history")
async def get_analysis_history(
    reasoning_model: Optional[str] = None,
    analysis_model: Optional[str] = None,
    limit: int = 100,
    user=Depends(get_verified_user),
) -> Dict[str, Any]:
    """
    Get list of available analysis history records.

    Can filter by reasoning model (the model being analyzed) and/or
    analysis model (the model used for analysis).
    """
    try:
        records = list_analysis_history(
            reasoning_model=reasoning_model,
            analysis_model=analysis_model,
            limit=min(limit, 500),
        )

        # Group by reasoning model for easier UI display
        by_model = {}
        for record in records:
            model = record.get("reasoning_model", "unknown")
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(record)

        return {
            "success": True,
            "records": records,
            "by_reasoning_model": by_model,
            "total": len(records),
        }
    except Exception as e:
        log.error("Failed to get analysis history: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/reasoning/report/generate")
async def generate_model_report(
    form_data: ReportGenerationRequest,
    request: Request,
    user=Depends(get_verified_user),
):
    """
    Generate an analysis report for a reasoning model.

    This endpoint aggregates analysis results from multiple conversations
    and generates a comprehensive report about the reasoning model's
    characteristics, error patterns, and recommendations.

    Supports streaming responses for real-time progress updates.
    """
    log.info(
        "Report generation requested | reasoning_model=%s analysis_model=%s report_model=%s",
        form_data.reasoning_model,
        form_data.analysis_model,
        form_data.report_model,
    )

    # Pre-fetch models list once to avoid repeated API calls to Ollama/OpenAI
    # This prevents connection errors when those services are unavailable
    cached_models = None

    async def get_cached_models():
        nonlocal cached_models
        if cached_models is None:
            try:
                cached_models = await get_all_models(request=request, user=user)
            except Exception as e:
                log.warning(
                    "Failed to fetch models list: %s. Will try direct model call.", e
                )
                cached_models = []
        return cached_models

    async def llm_call_func(messages, model, stream=False):
        """Call the LLM for report generation."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        # Use cached models list instead of fetching every time
        models = await get_cached_models()
        model_info = next((m for m in models if m.get("id") == model), None)

        # If model not found in cache but we have a model ID, try anyway
        # This allows the call to proceed even if model list fetch failed
        if not model_info and not models:
            log.warning(
                "Models list unavailable, proceeding with direct call to model: %s",
                model,
            )
        elif not model_info:
            raise ValueError(f"Model not found: {model}")

        form_data_dict = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        if stream:
            # For streaming, return an async generator directly (not a coroutine)
            response = await generate_chat_completion(
                request=request,
                form_data=form_data_dict,
                user=user,
            )

            async def stream_gen():
                if hasattr(response, "body_iterator"):
                    async for chunk in response.body_iterator:
                        if chunk:
                            chunk_str = (
                                chunk.decode("utf-8")
                                if isinstance(chunk, bytes)
                                else chunk
                            )
                            # Parse SSE format
                            for line in chunk_str.split("\n"):
                                if line.startswith("data: "):
                                    data = line[6:]
                                    if data.strip() == "[DONE]":
                                        continue
                                    try:
                                        parsed = json.loads(data)
                                        choices = parsed.get("choices", [])
                                        if choices and len(choices) > 0:
                                            delta = choices[0].get("delta", {})
                                            content = delta.get("content", "")
                                            if content:
                                                yield {"content": content}
                                        # Also handle error responses
                                        if "error" in parsed:
                                            log.error(
                                                "Streaming error: %s",
                                                parsed.get("error"),
                                            )
                                    except json.JSONDecodeError:
                                        continue
                else:
                    yield {"content": str(response)}

            # Return the generator object (not a coroutine)
            return stream_gen()
        else:
            # Non-streaming call
            response = await generate_chat_completion(
                request=request,
                form_data=form_data_dict,
                user=user,
            )
            return response

    async def stream_generator() -> AsyncGenerator[str, None]:
        """Generate SSE stream for report generation."""
        try:
            async for event in generate_analysis_report(
                reasoning_model=form_data.reasoning_model,
                analysis_model=form_data.analysis_model or "",
                report_model=form_data.report_model,
                llm_call_func=llm_call_func,
                directories=form_data.directories,
                stream=form_data.stream,
                language=form_data.language,
            ):
                event_type = event.get("type", "update")
                yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"

        except Exception as e:
            log.error("Report generation failed: %s", e)
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/reasoning/report/stats")
async def get_aggregated_stats(
    reasoning_model: Optional[str] = None,
    analysis_model: Optional[str] = None,
    limit: int = 100,
    user=Depends(get_verified_user),
) -> Dict[str, Any]:
    """
    Get aggregated statistics for a reasoning model without generating a full report.

    Useful for quick overview and previewing before generating a full report.
    """
    try:
        # Get analysis history
        history = list_analysis_history(
            reasoning_model=reasoning_model,
            analysis_model=analysis_model,
            limit=min(limit, 500),
        )

        if not history:
            return {
                "success": True,
                "stats": None,
                "message": "No analysis records found",
                "records_count": 0,
            }

        # Load full records
        records = []
        for item in history:
            record = load_full_analysis_record(item["directory"])
            if record:
                records.append(record)

        if not records:
            return {
                "success": True,
                "stats": None,
                "message": "Could not load analysis records",
                "records_count": 0,
            }

        # Aggregate stats
        stats = aggregate_analysis_stats(records)

        return {
            "success": True,
            "stats": stats.to_dict(),
            "records_count": len(records),
            "reasoning_model": reasoning_model,
            "analysis_model": analysis_model,
        }
    except Exception as e:
        log.error("Failed to get aggregated stats: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# =============================================================================
# Saved Reports Management API
# =============================================================================


@router.get("/reasoning/report/list")
async def list_reports(
    reasoning_model: Optional[str] = None,
    limit: int = 50,
    user=Depends(get_verified_user),
) -> Dict[str, Any]:
    """
    List all saved analysis reports.

    Returns a list of report summaries (without full content) for display in the UI.
    Reports are sorted by timestamp, newest first.
    """
    try:
        reports = list_saved_reports(
            reasoning_model=reasoning_model,
            limit=min(limit, 100),
        )

        return {
            "success": True,
            "reports": reports,
            "total": len(reports),
        }
    except Exception as e:
        log.error("Failed to list saved reports: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/reasoning/report/saved/{report_id}")
async def get_report(
    report_id: str,
    user=Depends(get_verified_user),
) -> Dict[str, Any]:
    """
    Get a specific saved report by ID.

    Returns the full report content including the generated markdown report.
    """
    try:
        report = get_saved_report(report_id)

        if report is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report not found: {report_id}",
            )

        return {
            "success": True,
            "report": report,
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to get saved report %s: %s", report_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.delete("/reasoning/report/saved/{report_id}")
async def delete_report(
    report_id: str,
    user=Depends(get_verified_user),
) -> Dict[str, Any]:
    """
    Delete a saved report by ID.
    """
    try:
        success = delete_saved_report(report_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report not found: {report_id}",
            )

        return {
            "success": True,
            "message": f"Report deleted: {report_id}",
        }
    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to delete report %s: %s", report_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
