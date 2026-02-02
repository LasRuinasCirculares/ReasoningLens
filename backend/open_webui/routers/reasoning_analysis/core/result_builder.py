"""
Result Builder Module

This module handles building the final merged analysis result
from Layer 1 and Layer 2 analysis outputs.
"""

import logging
from typing import Optional, List, Dict

from ..extractors.sections import get_section_range_text

log = logging.getLogger(__name__)


def build_final_result(
    layer1_json: dict,
    layer2_data: dict,
    sections: Optional[List[Dict]] = None,
) -> dict:
    """
    Build the final merged analysis result.

    Args:
        layer1_json: Layer 1 analysis result
        layer2_data: Layer 2 analysis results by node_id
        sections: Pre-split sections of the reasoning text

    Returns:
        Final merged analysis result with nodes, edges, and metadata
    """
    layer1_nodes = layer1_json.get("nodes", [])
    layer1_edges = layer1_json.get("edges", [])

    # Enrich Layer 1 nodes with Layer 2 data and segment positions
    enriched_nodes = []
    for node in layer1_nodes:
        node_id = node.get("id", "")
        node_type = node.get("type", "")

        # Build enriched node
        enriched_node = {
            "id": node_id,
            "type": node_type,
            "label": node.get("label", ""),
            "description": node.get("description", ""),
            "metadata": node.get("metadata", {}),
        }

        # Add section references
        if node.get("section_start") and node.get("section_end"):
            enriched_node["section_start"] = node.get("section_start")
            enriched_node["section_end"] = node.get("section_end")

        # Add segment boundary information (character positions)
        if "_segment_start" in node and "_segment_end" in node:
            enriched_node["segment_pos_start"] = node["_segment_start"]
            enriched_node["segment_pos_end"] = node["_segment_end"]

        # Add extracted segment text (the actual text from reasoning, not a summary)
        if "_segment_text" in node:
            enriched_node["segment_text"] = node["_segment_text"]
        elif sections and node.get("section_start") and node.get("section_end"):
            # Extract segment text from sections
            segment_text, _, _ = get_section_range_text(
                sections, node.get("section_start"), node.get("section_end")
            )
            if segment_text:
                enriched_node["segment_text"] = segment_text

        # Add Layer 2 data if available
        if node_id in layer2_data:
            node_layer2 = layer2_data[node_id]
            tree = node_layer2.get("tree", {"nodes": [], "edges": []})

            # Layer 2 now uses original section numbers directly
            steps = node_layer2.get("steps", [])
            tree_nodes = tree.get("nodes", [])
            tree_edges = tree.get("edges", [])

            tree = {
                "nodes": tree_nodes,
                "edges": tree_edges,
                "summary": tree.get("summary", ""),
            }

            can_be_refined = node_layer2.get(
                "can_be_refined", len(steps) > 0 or len(tree_nodes) > 0
            )

            enriched_node["layer2"] = {
                "steps": steps,
                "issues": node_layer2.get("issues", []),
                "can_be_refined": can_be_refined,
                "refinement_reason": node_layer2.get("refinement_reason", ""),
                "tree": tree,
            }
            enriched_node["has_layer2"] = True
            enriched_node["can_be_refined"] = can_be_refined
            # Also include the segment_text from layer2_data if not already present
            if "segment_text" not in enriched_node and "segment_text" in node_layer2:
                enriched_node["segment_text"] = node_layer2["segment_text"]
        else:
            enriched_node["has_layer2"] = False
            enriched_node["can_be_refined"] = False

        # Add error flags
        if node_id in layer2_data and layer2_data[node_id]["issues"]:
            enriched_node["issues"] = layer2_data[node_id]["issues"]
            enriched_node["has_error"] = True
        else:
            enriched_node["issues"] = []
            enriched_node["has_error"] = False

        enriched_nodes.append(enriched_node)

    # Build edge list with labels
    enriched_edges = []
    for edge in layer1_edges:
        enriched_edge = {
            "from": edge.get("from", ""),
            "to": edge.get("to", ""),
            "type": edge.get("type", "normal"),
            "label": edge.get("label", ""),
        }
        enriched_edges.append(enriched_edge)

    # Build sections array for frontend (include position info for accurate highlighting)
    sections_for_output = []
    if sections:
        for section in sections:
            sections_for_output.append(
                {
                    "section_num": section.get("section_id", 0),
                    "text": section.get("text", ""),
                    "start_pos": section.get("start_pos", 0),
                    "end_pos": section.get("end_pos", 0),
                }
            )

    # Build final payload
    return {
        "summary": layer1_json.get("summary", ""),
        "nodes": enriched_nodes,
        "edges": enriched_edges,
        "sections": sections_for_output,  # Include sections for frontend highlighting
        "analysis_metadata": {
            "layer1_node_count": len(layer1_nodes),
            "layer2_node_count": len(layer2_data),
            "total_layer2_steps": sum(len(d["steps"]) for d in layer2_data.values()),
            "total_layer2_tree_nodes": sum(
                len((d.get("tree") or {}).get("nodes", []))
                for d in layer2_data.values()
            ),
            "total_issues": sum(len(d["issues"]) for d in layer2_data.values()),
            "node_type_distribution": {
                node_type: sum(1 for n in layer1_nodes if n.get("type") == node_type)
                for node_type in set(n.get("type") for n in layer1_nodes)
            },
            "sections_count": len(sections_for_output),
        },
    }
