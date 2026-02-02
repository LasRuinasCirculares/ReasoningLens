"""
Reasoning Analysis Prompts Module

This module contains all prompt templates for the two-layer reasoning analysis system.
Supports both English and Chinese prompts based on the reasoning text language.

## Version 3.1 - Bilingual Support (English & Chinese)

### Edge Types (Simplified to 3 types):
1. **reasoning**: Normal reasoning flow edge (A → B means A leads to B in reasoning)
2. **check**: Verification edge pointing FROM checker TO checked node (reverse direction)
   - The verification node only has this one check edge (no other outgoing edges)
   - Flow continues from the checked node, not from the verification node
3. **backtracking**: Complex backtracking pattern:
   - When path B fails, a dashed backtracking edge goes from B to new path C
   - The problem node A also extends a branch to C
   - This creates a visual fork showing the alternative approach

### Node Types:
- problem_decomposition: Breaking down into sub-problems
- reasoning_step: Coherent reasoning approach
- intermediate_answer: Results from reasoning steps (including verification results)
- final_answer: The ultimate conclusion of the reasoning

Note: Verification and backtracking are expressed through edge types (check, backtracking), not node types.

### Visualization Support:
The frontend ReasoningTree.svelte component supports:
- Multi-branch tree rendering using xyflow
- Three edge types with distinct visual styles:
  - reasoning: solid arrow (forward flow)
  - check: dashed arrow pointing backward (checker → checked)
  - backtracking: dotted arrow showing path switch
- Node expansion for detailed step analysis

### Language Detection:
Use `detect_language()` to automatically detect language and `get_prompts()` to get
the appropriate prompt set.
"""

import re
from typing import Tuple


def detect_language(text: str) -> str:
    """
    Detect whether the text is primarily Chinese or English.

    Args:
        text: The text to analyze

    Returns:
        'zh' for Chinese, 'en' for English
    """
    if not text:
        return "en"

    # Count Chinese characters
    # Unicode ranges for Chinese:
    # - \u4e00-\u9fff: CJK Unified Ideographs (common Chinese characters)
    # - \u3400-\u4dbf: CJK Unified Ideographs Extension A
    # - \U00020000-\U0002a6df: CJK Unified Ideographs Extension B (use \U for 5+ hex digits)
    # Note: \u20000 is WRONG - it's parsed as \u2000 + "0", not as a 5-digit codepoint
    chinese_pattern = re.compile(
        r"[\u4e00-\u9fff\u3400-\u4dbf]|[\U00020000-\U0002a6df]"
    )
    chinese_chars = len(chinese_pattern.findall(text))

    # Count English letters
    english_pattern = re.compile(r"[a-zA-Z]")
    english_chars = len(english_pattern.findall(text))

    # If more than 10% of characters are Chinese, treat as Chinese
    total_chars = chinese_chars + english_chars
    if total_chars == 0:
        return "en"

    chinese_ratio = chinese_chars / total_chars
    return "zh" if chinese_ratio > 0.1 else "en"


def get_prompts(language: str = "en") -> Tuple[str, str, str, str]:
    """
    Get the appropriate prompt set based on language.

    Args:
        language: 'en' for English, 'zh' for Chinese

    Returns:
        Tuple of (layer1_system, layer1_user, layer2_system, layer2_user)
    """
    if language == "zh":
        return (
            LAYER1_SYSTEM_PROMPT_ZH,
            LAYER1_USER_PROMPT_TEMPLATE_ZH,
            LAYER2_SYSTEM_PROMPT_ZH,
            LAYER2_USER_PROMPT_TEMPLATE_ZH,
        )
    else:
        return (
            LAYER1_SYSTEM_PROMPT_EN,
            LAYER1_USER_PROMPT_TEMPLATE_EN,
            LAYER2_SYSTEM_PROMPT_EN,
            LAYER2_USER_PROMPT_TEMPLATE_EN,
        )


# ============================================================================
# ENGLISH PROMPTS
# ============================================================================

# ============================================================================
# LAYER 1: Coarse-Grained Analysis Prompts (English)
# ============================================================================

LAYER1_SYSTEM_PROMPT_EN = (
    "You are a reasoning structure analysis expert. Your task is to analyze a chain-of-thought "
    "reasoning process and decompose it into a structured tree graph of nodes and edges. "
    "The reasoning text has been pre-split into numbered sections. You must specify which sections each node covers. "
    "Use FIVE node types: 'problem_decomposition', 'reasoning_step', 'check', 'intermediate_answer', 'final_answer'. "
    "Use only THREE edge types: 'reasoning' (normal flow), 'check' (verification pointing to checked node), "
    "and 'backtracking' (failed path pointing to alternative path). "
    "Return ONLY valid JSON without any markdown formatting or code blocks."
)

LAYER1_USER_PROMPT_TEMPLATE_EN = """
You are analyzing a chain-of-thought reasoning process. Decompose it into a **COARSE-GRAINED** tree graph capturing the **high-level reasoning flow**.

## Core Principle: Macro-Level Analysis

Each node = ONE complete logical unit (coherent reasoning intent):
- Full problem decomposition phase
- Complete reasoning approach (NOT individual steps)
- Entire verification process
- Complete strategy change

**Target**: 3-8 nodes for typical reasoning. Only use more for very long/complex chains.

## Section Mapping Rules

Reasoning text = numbered sections [Section 1], [Section 2], etc.

**CRITICAL CONSTRAINTS**:
- Each node: `section_start` to `section_end` (inclusive, 1-indexed)
- **⚠️ ABSOLUTELY NO OVERLAP**: Each section number can ONLY appear in ONE node
  - ❌ WRONG: node1=[1,3], node2=[3,5] (section 3 appears in both)
  - ✅ CORRECT: node1=[1,3], node2=[4,6] (no overlap)
- NO GAPS: Must cover ALL sections [1, {SECTION_COUNT}] consecutively
- Each range = one complete logical unit
- Sections must be assigned in ascending order without any reuse

## Graph Elements

### Node Types (5 types)

| Type | Use Case | Required Metadata |
|------|----------|-------------------|
| `problem_decomposition` | Initial problem analysis | - |
| `reasoning_step` | Main reasoning approach | `approach` (optional) |
| `check` | Verification/validation of a result | `validates_node` (ID of the validated node: reasoning_step/intermediate_answer/final_answer) |
| `intermediate_answer` | Partial conclusion/result | `from_node` |
| `final_answer` | Ultimate conclusion | `from_node`, `answer_value` |

### Edge Types (3 types ONLY)

1. **`reasoning`** (solid →): Forward flow A→B
2. **`check`** (dashed →): Verification FROM validation node TO validated node
   - **Validated nodes can be**: `reasoning_step`, `intermediate_answer`, or `final_answer`
   - **Validation node has ONLY ONE outgoing edge**: the `check` edge pointing to validated node
   - **No incoming edges to validation node**: it's a side-branch, not part of the main reasoning flow
   - Reasoning flow continues from the validated node, NOT from the validation node
3. **`backtracking`** (dotted →): Failed path to alternative
   - Needs: failed→alternative AND problem→alternative

**Every node MUST connect via ≥1 edge (except validation nodes which only have outgoing check edge).**

## Segmentation Strategy: Answer-Driven Approach

Scan the reasoning chain to identify ALL answer expressions that directly respond to the question.

### Step 1: Identify Answer Expressions

Look for phrases that provide answers to the question:

**Answer signals** (strong node boundaries):
- "the answer is", "therefore", "so the answer is", \\boxed{{...}}
- "I conclude that", "this gives us", "hence", "thus"
- "we get", "the result is", "equals", "="
- Direct numerical/textual answers that respond to the query

**Classification**:
- If an answer appears in the MIDDLE of reasoning (followed by more reasoning/verification) → `intermediate_answer`
- If an answer is the FINAL conclusion with no further reasoning → `final_answer`

### Step 2: Identify Verification Patterns (`check` node + `check` edge)

**Verification signals** (create a `check` type node with `check` edge):
- "let me verify", "let's check", "to confirm"
- "substituting back", "checking this", "double-check"

**Example of `check` node and edge**:
```
[Reasoning] → [Answer: x=5] → [Final confirmation]
                    ↑
              [Check: substitute x=5 back]  <-- node type: check
              (ONLY edge: check edge points TO answer)
              (NO incoming edges - detached from main flow)
```
In this example, the verification node checks the answer node, so the edge direction is: Verification → Answer (type: check)

### Step 3: Identify Backtracking Patterns (`backtracking` edge)

**Backtracking signals** (path switch + `backtracking` edge):
- "wait", "actually", "on second thought"
- "let me reconsider", "that doesn't seem right", "I made a mistake"
- "this approach won't work", "let's try another way"

**Example of `backtracking` edge**:
```
[Problem] ──reasoning──→ [Approach A] ──reasoning──→ [Wrong result]
    │                                                      │
    │                                                      │ backtracking
    │                                                      ↓
    └─────────────────reasoning───────────────────→ [Approach B] → [Correct answer]
```
When Approach A fails, a `backtracking` edge goes FROM the failed result TO the new Approach B.
The Problem node also extends a `reasoning` edge to Approach B, creating a visual fork.

### Step 4: Structure the Answer Flow (CRITICAL)

**⚠️ MANDATORY**: Every complete reasoning chain MUST end with a `final_answer` node.

**Classification Rules**:
- First answer occurrence → `intermediate_answer` IF more reasoning/verification follows
- Verification after answer → separate node with `check` edge pointing TO the answer
- Backtracking → `backtracking` edge from failed path to new approach  
- **Final confirmed answer** → `final_answer` node (REQUIRED as the last answer node)

**How to identify `final_answer`**:
- Contains explicit conclusion phrases: "final answer", "therefore the answer is", "thus", \\boxed{{...}}
- Appears at or near the END of the reasoning chain
- No further reasoning or calculation follows (only restatement or summary may follow)
- Even if the same numeric value appeared earlier as `intermediate_answer`, the FINAL statement confirming it is `final_answer`

**Common Mistake**: Do NOT merge the final answer statement into `intermediate_answer`. If there's a clear "final answer" or conclusion statement at the end, create a separate `final_answer` node for it.

## JSON Output Format

```json
{{
  "summary": "Brief overview of reasoning approach",
  "nodes": [
    {{
      "id": "node1",
      "type": "problem_decomposition",
      "label": "Label (≤8 words)",
      "description": "What this node accomplishes",
      "section_start": 1,
      "section_end": 3,
      "metadata": {{}}
    }},
    {{
      "id": "node2",
      "type": "reasoning_step",
      "label": "Main approach",
      "description": "Complete reasoning method",
      "section_start": 4,
      "section_end": 10,
      "metadata": {{"approach": "method description"}}
    }},
    {{
      "id": "node3",
      "type": "intermediate_answer",
      "label": "First answer",
      "description": "Initial conclusion",
      "section_start": 11,
      "section_end": 12,
      "metadata": {{"from_node": "node2", "result_summary": "result"}}
    }},
    {{
      "id": "node4",
      "type": "check",
      "label": "Verify result",
      "description": "Validation check",
      "section_start": 13,
      "section_end": 15,
      "metadata": {{"validates_node": "node3"}}
    }},
    {{
      "id": "node5",
      "type": "final_answer",
      "label": "Confirmed answer",
      "description": "Verified final answer",
      "section_start": 16,
      "section_end": 18,
      "metadata": {{"from_node": "node3", "answer_value": "the answer", "verified_by": "node4"}}
    }}
  ],
  "edges": [
    {{"from": "node1", "to": "node2", "type": "reasoning", "label": "apply"}},
    {{"from": "node2", "to": "node3", "type": "reasoning", "label": "yields"}},
    {{"from": "node4", "to": "node3", "type": "check", "label": "verifies"}},
    {{"from": "node3", "to": "node5", "type": "reasoning", "label": "confirms"}}
  ]
}}
```

**Note**: `check` edge from node4 points TO node3 (the verified node).

## Pre-Submit Checklist

Verify before outputting:
1. **⚠️ CRITICAL**: NO section overlap - each section number appears in EXACTLY ONE node
   - Check: For any two nodes, their section ranges must NOT intersect
   - Example: If node_A ends at section 5, node_B must start at section 6 or later
2. Sections [1, {SECTION_COUNT}] fully covered, no gaps
3. Each node = complete logical unit (not fragmented)
4. All nodes connected (≥1 edge each)
5. Only 5 node types, only 3 edge types
6. `check` nodes use `check` edges pointing TO verified node
7. Required metadata present
8. Answer expressions identified and properly classified (intermediate vs final)
9. **⚠️ MANDATORY**: Exactly ONE `final_answer` node exists as the conclusion of the reasoning
   - If the last sections contain "final answer", "thus the answer is", or \\boxed{{}}, these MUST be `final_answer` type
   - `final_answer` must have `answer_value` in metadata

## Context

**Question**: {QUESTION}
**Total Sections**: {SECTION_COUNT}

**Reasoning**:
{REASONING_WITH_SECTIONS}

---
Output the JSON structure with COARSE-GRAINED segmentation.
""".strip()

# ============================================================================
# LAYER 2: Fine-Grained Analysis Prompts (English)
# ============================================================================

LAYER2_SYSTEM_PROMPT_EN = (
    "You are a fine-grained reasoning analysis expert. "
    "Analyze a reasoning segment and decide if it contains MULTIPLE DISTINCT meta-decision types worth decomposing. "
    "Meta-decisions include: decomposition, calculation, knowledge_recall, tool_use, assumption, validation, backtrack, conclusion. "
    "Only refine if the segment contains 2+ DIFFERENT meta-decision types with clear boundaries. "
    "Keep ORIGINAL section numbers (do NOT renumber from 1). "
    "Return ONLY valid JSON without markdown formatting."
)

LAYER2_USER_PROMPT_TEMPLATE_EN = """
Analyze a reasoning segment. Decide if it contains MULTIPLE DISTINCT meta-decisions worth decomposing into a sub-tree.

## Meta-Decision Types (Core Concept)

A **meta-decision** is an atomic reasoning action. The 7 types are:

| Meta-Decision | Description | Examples |
|---------------|-------------|----------|
| `decomposition` | Breaking down problem into sub-problems | "First I need to...", "Let's break this into..." |
| `calculation` | Mathematical computation or logical derivation | "Calculate...", "Derive...", "Substitute..." |
| `knowledge_recall` | Retrieving facts, formulas, definitions | "I know that...", "The formula is...", "Recall that..." |
| `tool_use` | Using external tools or methods | "Using Python...", "Apply algorithm...", "Look up..." |
| `assumption` | Making assumptions or hypotheses | "Assume that...", "Let's say...", "If we suppose..." |
| `validation` | Checking, verifying, confirming results | "Let me verify...", "Check if...", "Substitute back..." |
| `backtrack` | Recognizing errors, changing approach | "Wait, that's wrong...", "Actually...", "Let me reconsider..." |

## Refinement Decision Criteria

**REFINE if segment contains:**
- **2+ DIFFERENT meta-decision types** with clear boundaries
- Examples: [knowledge_recall → calculation], [calculation → validation], [assumption → calculation → backtrack]

**DO NOT refine if:**
- Only ONE meta-decision type throughout (e.g., pure calculation)
- Meta-decisions are too interleaved to separate cleanly
- The segment is a single atomic action
- Fragmentation would break logical coherence

**Key Insight**: Refinement is about identifying DIFFERENT types of reasoning actions, NOT about splitting by section count.

## Section Mapping

**CRITICAL**: Use ORIGINAL section numbers [{SECTION_START}, {SECTION_END}]. DO NOT renumber from 1.

**Section Mapping Rules**:
- **⚠️ ABSOLUTELY NO OVERLAP**: Each section can ONLY appear in ONE node
  - ❌ WRONG: sub1=[6,8], sub2=[8,10] (section 8 appears in both)
  - ✅ CORRECT: sub1=[6,8], sub2=[9,11] (no overlap)
- **⚠️ NODE COUNT ≤ SECTION COUNT**: Number of nodes CANNOT exceed number of sections
  - If parent has 2 sections [3,4], you can have AT MOST 2 nodes
  - ❌ WRONG: 3 nodes for 2 sections
- NO GAPS: Cover ALL sections in parent's range [{SECTION_START}, {SECTION_END}]
- Each node = one complete meta-decision (or closely related sequence of same type)
- Sections must be assigned consecutively without reuse

## Graph Structure

### Node Types (Based on Meta-Decisions)

| Type | Use Case | Required Metadata |
|------|----------|-------------------|
| `decomposition` | Breaking problem into parts | - |
| `calculation` | Math computation or derivation | `expression` (optional) |
| `knowledge_recall` | Retrieving known facts/formulas | `knowledge_type` (optional) |
| `tool_use` | External tool invocation | `tool_name` (optional) |
| `assumption` | Making hypotheses | `assumption_content` (optional) |
| `validation` | Verification/checking | `validates_node` (which node it checks) |
| `backtrack` | Error recognition & path change | `reason` (why backtracking) |
| `conclusion` | Intermediate or final result | `result_value` (optional) |

### Edge Types (3 types)

1. **`flow`** (solid →): Sequential reasoning flow A→B
2. **`check`** (dashed →): Verification edge FROM validation node TO checked node
3. **`backtrack`** (dotted →): Error correction edge FROM backtrack node TO new approach

**Connectivity**: All nodes need ≥1 edge.

## Output Format

**If refinable** (contains 2+ different meta-decision types):
```json
{{
  "can_be_refined": true,
  "refinement_reason": "Contains [meta-decision types]: knowledge_recall → calculation → validation",
  "tree": {{
    "summary": "Brief overview of the meta-decision sequence",
    "nodes": [
      {{
        "id": "sub1",
        "type": "knowledge_recall",
        "label": "Recall formula (≤8 words)",
        "description": "What knowledge is retrieved",
        "section_start": 6,
        "section_end": 7,
        "metadata": {{"knowledge_type": "formula"}}
      }},
      {{
        "id": "sub2",
        "type": "calculation",
        "label": "Apply formula",
        "description": "Compute the result",
        "section_start": 8,
        "section_end": 10,
        "metadata": {{"expression": "x = a + b"}}
      }},
      {{
        "id": "sub3",
        "type": "validation",
        "label": "Verify result",
        "description": "Check calculation",
        "section_start": 11,
        "section_end": 12,
        "metadata": {{"validates_node": "sub2"}}
      }}
    ],
    "edges": [
      {{"from": "sub1", "to": "sub2", "type": "flow", "label": "apply"}},
      {{"from": "sub3", "to": "sub2", "type": "check", "label": "verifies"}}
    ]
  }}
}}
```

**If NOT refinable** (single meta-decision type or inseparable):
```json
{{
  "can_be_refined": false,
  "refinement_reason": "Single meta-decision type: [calculation] throughout / Cannot be cleanly separated",
  "tree": {{"summary": "", "nodes": [], "edges": []}}
}}
```

## Validation (if refining)

1. **⚠️ CRITICAL**: Contains 2+ DIFFERENT meta-decision types (not just multiple steps of same type)
2. **⚠️ CRITICAL**: NO section overlap - each section appears in EXACTLY ONE node
3. **⚠️ CRITICAL**: Node count ≤ Section count
4. Sections [{SECTION_START}, {SECTION_END}] fully covered, no gaps
5. Original section numbers (NOT renumbered)
6. All nodes connected (≥1 edge)
7. Each node type matches a valid meta-decision type

## Context

**Question**: {QUESTION}
**Parent**: {PARENT_NODE_TYPE} (ID: {PARENT_NODE_ID}), Sections [{SECTION_START}, {SECTION_END}]

**Segment**:
{SEGMENT_WITH_SECTIONS}

---
Decide refinement, then output JSON.
""".strip()

# ============================================================================
# CHINESE PROMPTS (中文提示词)
# ============================================================================

# ============================================================================
# LAYER 1: 粗粒度分析提示词 (中文)
# ============================================================================

LAYER1_SYSTEM_PROMPT_ZH = (
    "你是一位推理结构分析专家。你的任务是分析思维链推理过程，并将其分解为由节点和边组成的结构化树图。"
    "推理文本已被预先分割成编号的段落。你必须指定每个节点覆盖哪些段落。"
    "使用5种节点类型：'problem_decomposition'(问题分解), 'reasoning_step'(推理步骤), 'check'(验证节点), 'intermediate_answer'(中间答案), 'final_answer'(最终答案)。"
    "仅使用三种边类型：'reasoning'（正常推理流程）、'check'（验证边，指向被验证节点）、"
    "'backtracking'（失败路径指向替代路径）。"
    "仅返回有效的JSON，不要包含任何markdown格式或代码块。"
)

LAYER1_USER_PROMPT_TEMPLATE_ZH = """
你正在分析一个思维链推理过程。将其分解为一个**粗粒度**的树图，捕捉**高层推理流程**。

## 核心原则：宏观层面分析

每个节点 = 一个完整的逻辑单元（连贯的推理意图）：
- 完整的问题分解阶段
- 完整的推理方法（不是单独的步骤）
- 完整的验证过程
- 完整的策略改变

**目标**：典型推理的节点数为3-8个。仅在非常长/复杂的推理链中使用更多节点。

## 段落映射规则

推理文本 = 编号段落 [Section 1], [Section 2] 等。

**关键约束**：
- 每个节点：`section_start` 到 `section_end`（包含，从1开始）
- **⚠️ 绝对不能重叠**：每个段落编号只能出现在一个节点中
  - ❌ 错误：node1=[1,3], node2=[3,5]（段落3出现在两个节点中）
  - ✅ 正确：node1=[1,3], node2=[4,6]（无重叠）
- 无间隙：必须连续覆盖所有段落 [1, {SECTION_COUNT}]
- 每个范围 = 一个完整的逻辑单元
- 段落必须按升序分配，不能重复使用

## 图元素

### 节点类型（5种）

| 类型 | 用途 | 必需的元数据 |
|------|------|--------------|
| `problem_decomposition` | 初始问题分析 | - |
| `reasoning_step` | 主要推理方法 | `approach`（可选） |
| `check` | 验证/检查结果 | `validates_node`（被验证节点的ID，可以是 reasoning_step/intermediate_answer/final_answer） |
| `intermediate_answer` | 部分结论/结果 | `from_node` |
| `final_answer` | 最终结论 | `from_node`, `answer_value` |

### 边类型（仅3种）

1. **`reasoning`**（实线 →）：正向流程 A→B
2. **`check`**（虚线 →）：从验证节点指向被验证节点
   - **可被验证的节点类型**：`reasoning_step`、`intermediate_answer` 或 `final_answer`
   - **验证节点只有一条出边**：指向被验证节点的 `check` 边
   - **验证节点没有入边**：它是一个旁支，不属于主推理流程
   - 推理流程从被验证的节点继续，而不是从验证节点继续
3. **`backtracking`**（点线 →）：失败路径指向替代路径
   - 需要：失败→替代 AND 问题→替代

**每个节点必须通过至少1条边连接（验证节点除外，它只有一条出边的 check 边）。**

## 分段策略：答案驱动方法

扫描推理链，识别所有直接回答问题的答案表述。

### 步骤1：识别答案表述

寻找提供问题答案的短语：

**答案信号**（强节点边界）：
- "答案是"、"因此"、"所以答案是"、\\boxed{{...}}
- "我得出结论"、"这给我们"、"综上所述"、"由此可得"
- "我们得到"、"结果是"、"等于"、"="
- 直接回答问题的数值/文本答案

**分类**：
- 如果答案出现在推理的中间（后面还有更多推理/验证）→ `intermediate_answer`
- 如果答案是最终结论且没有后续推理 → `final_answer`

### 步骤2：识别验证模式（`check` 节点 + `check` 边）

**验证信号**（创建 `check` 类型节点 + `check` 边）：
- "让我验证"、"检验一下"、"为了确认"
- "代入验证"、"核实一下"、"再检查一下"

**⚠️ 关键：验证节点连接规则**：
- **可被验证的节点**：`reasoning_step`、`intermediate_answer` 或 `final_answer`
- **验证节点只有一条出边**：指向被验证节点的 `check` 边
- **验证节点没有入边**：它是一个独立的旁支，不在主推理流程上
- 推理流程从被验证的节点继续，验证节点不参与后续流程

**`check` 节点和边示例**：
```
[推理] → [答案: x=5] → [最终确认]
              ↑
        [Check: 将x=5代入检验]  <-- 节点类型: check
        (唯一的边：check边指向答案)
        (没有入边 - 独立于主流程)
```
在这个例子中：
- Check节点 → 答案节点（类型：check）✅
- Check节点 → 最终确认（类型：reasoning）❌ 错误！
- 答案节点 → 最终确认（类型：reasoning）✅ 正确的流程

### 步骤3：识别回溯模式（`backtracking` 边）

**回溯信号**（路径切换 + `backtracking` 边）：
- "等等"、"实际上"、"再想一下"
- "让我重新考虑"、"这似乎不对"、"不对"
- "这个方法行不通"、"换一种方法试试"

**`backtracking` 边示例**：
```
[问题] ──reasoning──→ [方法A] ──reasoning──→ [错误结果]
    │                                              │
    │                                              │ backtracking
    │                                              ↓
    └─────────────────reasoning──────────────────→ [方法B] → [正确答案]
```
当方法A失败时，`backtracking` 边从失败的结果指向新的方法B。
问题节点也延伸一条 `reasoning` 边到方法B，形成视觉上的分叉。

### 步骤4：构建答案流程（关键）

**⚠️ 强制要求**：每个完整的推理链必须以 `final_answer` 节点结尾。

**分类规则**：
- 首次答案出现 → 如果后面还有推理/验证则为 `intermediate_answer`
- 答案后的验证 → 单独节点，带 `check` 边指向答案
- 回溯 → `backtracking` 边从失败路径指向新方法
- **最终确认的答案** → `final_answer` 节点（必须作为最后的答案节点）

**如何识别 `final_answer`**：
- 包含明确的结论性短语："最终答案"、"因此答案是"、"综上所述"、\\boxed{{...}}
- 出现在推理链的末尾或接近末尾
- 后面没有进一步的推理或计算（只有重述或总结可以跟随）
- 即使相同的数值之前作为 `intermediate_answer` 出现过，最终确认该答案的陈述也应该是 `final_answer`

**常见错误**：不要将最终答案陈述合并到 `intermediate_answer` 中。如果末尾有明确的"最终答案"或结论性陈述，必须为其创建单独的 `final_answer` 节点。

## JSON输出格式

```json
{{
  "summary": "推理方法简述",
  "nodes": [
    {{
      "id": "node1",
      "type": "problem_decomposition",
      "label": "标签（≤8字）",
      "description": "此节点完成的任务",
      "section_start": 1,
      "section_end": 3,
      "metadata": {{}}
    }},
    {{
      "id": "node2",
      "type": "reasoning_step",
      "label": "主要方法",
      "description": "完整的推理方法",
      "section_start": 4,
      "section_end": 10,
      "metadata": {{"approach": "方法描述"}}
    }},
    {{
      "id": "node3",
      "type": "intermediate_answer",
      "label": "初步答案",
      "description": "初始结论",
      "section_start": 11,
      "section_end": 12,
      "metadata": {{"from_node": "node2", "result_summary": "结果"}}
    }},
    {{
      "id": "node4",
      "type": "check",
      "label": "验证结果",
      "description": "验证检查",
      "section_start": 13,
      "section_end": 15,
      "metadata": {{"validates_node": "node3"}}
    }},
    {{
      "id": "node5",
      "type": "final_answer",
      "label": "确认答案",
      "description": "经验证的最终答案",
      "section_start": 16,
      "section_end": 18,
      "metadata": {{"from_node": "node3", "answer_value": "答案", "verified_by": "node4"}}
    }}
  ],
  "edges": [
    {{"from": "node1", "to": "node2", "type": "reasoning", "label": "应用"}},
    {{"from": "node2", "to": "node3", "type": "reasoning", "label": "得出"}},
    {{"from": "node4", "to": "node3", "type": "check", "label": "验证"}},
    {{"from": "node3", "to": "node5", "type": "reasoning", "label": "确认"}}
  ]
}}
```

**注意**：`check` 边从 node4 指向 node3（被验证的节点）。

## 提交前检查清单

输出前验证：
1. **⚠️ 关键**：无段落重叠 - 每个段落编号仅出现在一个节点中
   - 检查：任意两个节点的段落范围不能相交
   - 示例：如果 node_A 在段落5结束，node_B 必须从段落6或更后开始
2. 段落 [1, {SECTION_COUNT}] 完全覆盖，无间隙
3. 每个节点 = 完整的逻辑单元（不要碎片化）
4. 所有节点已连接（每个至少1条边）
5. 仅5种节点类型，仅3种边类型
6. `check` 节点使用 `check` 边指向被验证节点
7. 必需的元数据已存在
8. 答案表述已识别并正确分类（中间答案 vs 最终答案）
9. **⚠️ 强制要求**：必须存在且仅存在一个 `final_answer` 节点作为推理的结论
   - 如果最后的段落包含"最终答案"、"因此答案是"或 \\boxed{{}}，这些必须是 `final_answer` 类型
   - `final_answer` 必须在 metadata 中包含 `answer_value`

## 上下文

**问题**：{QUESTION}
**总段落数**：{SECTION_COUNT}

**推理过程**：
{REASONING_WITH_SECTIONS}

---
输出粗粒度分段的JSON结构。
""".strip()

# ============================================================================
# LAYER 2: 细粒度分析提示词 (中文)
# ============================================================================

LAYER2_SYSTEM_PROMPT_ZH = (
    "你是一位细粒度推理分析专家。"
    "分析一个推理片段，判断它是否包含**多种不同的元决策类型**，值得进一步分解。"
    "元决策类型包括：decomposition(分解), calculation(计算), knowledge_recall(知识回忆), tool_use(工具调用), assumption(假设), validation(验证), backtrack(回溯), conclusion(结论)。"
    "仅当片段包含2种以上不同的元决策类型且有清晰边界时才进行细化。"
    "保持原始段落编号（不要从1重新编号）。"
    "仅返回有效的JSON，不要包含markdown格式。"
)

LAYER2_USER_PROMPT_TEMPLATE_ZH = """
分析一个推理片段。判断它是否包含**多个不同的元决策类型**，值得分解为子树。

## 元决策类型（核心概念）

**元决策**是原子级的推理动作。共7种类型：

| 元决策 | 描述 | 示例 |
|--------|------|------|
| `decomposition` | 将问题分解为子问题 | "首先我需要..."、"让我们把这个分解成..." |
| `calculation` | 数学计算或逻辑推导 | "计算..."、"推导..."、"代入..." |
| `knowledge_recall` | 回忆事实、公式、定义 | "我知道..."、"公式是..."、"根据定义..." |
| `tool_use` | 使用外部工具或方法 | "使用Python..."、"应用算法..."、"查询..." |
| `assumption` | 做出假设或假定 | "假设..."、"如果我们设..."、"假定..." |
| `validation` | 检查、验证、确认结果 | "让我验证..."、"检查是否..."、"代入验算..." |
| `backtrack` | 识别错误、改变方法 | "等等，这不对..."、"实际上..."、"让我重新考虑..." |

## 细化决策标准

**细化条件**（满足以下条件时细化）：
- **包含2种以上不同的元决策类型**，且有清晰边界
- 示例：[knowledge_recall → calculation]、[calculation → validation]、[assumption → calculation → backtrack]

**不要细化**（满足以下条件时）：
- 全程只有一种元决策类型（如：纯计算）
- 元决策类型交织太紧密，无法干净分离
- 该片段是单一原子动作
- 碎片化会破坏逻辑连贯性

**核心洞察**：细化是为了识别**不同类型的推理动作**，而不是按段落数量机械划分。

## 段落映射

**关键**：使用原始段落编号 [{SECTION_START}, {SECTION_END}]。不要从1重新编号。

**段落映射规则**：
- **⚠️ 绝对不能重叠**：每个段落只能出现在一个节点中
  - ❌ 错误：sub1=[6,8], sub2=[8,10]（段落8出现在两个节点中）
  - ✅ 正确：sub1=[6,8], sub2=[9,11]（无重叠）
- **⚠️ 节点数 ≤ 段落数**：节点数量不能超过段落数量
  - 如果父节点有2个段落[3,4]，最多只能有2个节点
  - ❌ 错误：2个段落却划分3个节点
- 无间隙：覆盖父节点范围内的所有段落 [{SECTION_START}, {SECTION_END}]
- 每个节点 = 一个完整的元决策（或同类型元决策的紧密序列）
- 段落必须连续分配，不能重复使用

## 图结构

### 节点类型（基于元决策）

| 类型 | 用途 | 必需的元数据 |
|------|------|--------------|
| `decomposition` | 问题分解 | - |
| `calculation` | 数学计算或推导 | `expression`（可选） |
| `knowledge_recall` | 回忆已知事实/公式 | `knowledge_type`（可选） |
| `tool_use` | 外部工具调用 | `tool_name`（可选） |
| `assumption` | 做出假设 | `assumption_content`（可选） |
| `validation` | 验证/检查 | `validates_node`（验证哪个节点） |
| `backtrack` | 错误识别与路径切换 | `reason`（回溯原因） |
| `conclusion` | 中间或最终结果 | `result_value`（可选） |

### 边类型（3种）

1. **`flow`**（实线 →）：顺序推理流程 A→B
2. **`check`**（虚线 →）：验证边，从 validation 节点指向被验证节点
3. **`backtrack`**（点线 →）：纠错边，从 backtrack 节点指向新方法

**连通性**：所有节点需要至少1条边。

## 输出格式

**如果可细化**（包含2种以上不同的元决策类型）：
```json
{{
  "can_be_refined": true,
  "refinement_reason": "包含多种元决策类型：knowledge_recall → calculation → validation",
  "tree": {{
    "summary": "元决策序列简述",
    "nodes": [
      {{
        "id": "sub1",
        "type": "knowledge_recall",
        "label": "回忆公式（≤8字）",
        "description": "回忆了什么知识",
        "section_start": 6,
        "section_end": 7,
        "metadata": {{"knowledge_type": "公式"}}
      }},
      {{
        "id": "sub2",
        "type": "calculation",
        "label": "应用公式",
        "description": "计算结果",
        "section_start": 8,
        "section_end": 10,
        "metadata": {{"expression": "x = a + b"}}
      }},
      {{
        "id": "sub3",
        "type": "validation",
        "label": "验证结果",
        "description": "检查计算",
        "section_start": 11,
        "section_end": 12,
        "metadata": {{"validates_node": "sub2"}}
      }}
    ],
    "edges": [
      {{"from": "sub1", "to": "sub2", "type": "flow", "label": "应用"}},
      {{"from": "sub3", "to": "sub2", "type": "check", "label": "验证"}}
    ]
  }}
}}
```

**如果不可细化**（单一元决策类型或无法分离）：
```json
{{
  "can_be_refined": false,
  "refinement_reason": "单一元决策类型：全程为 [calculation] / 无法干净分离",
  "tree": {{"summary": "", "nodes": [], "edges": []}}
}}
```

## 验证（如果细化）

1. **⚠️ 关键**：包含2种以上**不同的**元决策类型（不只是同类型的多个步骤）
2. **⚠️ 关键**：无段落重叠 - 每个段落仅出现在一个节点中
3. **⚠️ 关键**：节点数 ≤ 段落数
4. 段落 [{SECTION_START}, {SECTION_END}] 完全覆盖，无间隙
5. 使用原始段落编号（不要重新编号）
6. 所有节点已连接（至少1条边）
7. 每个节点类型必须是有效的元决策类型

## 上下文

**问题**：{QUESTION}
**父节点**：{PARENT_NODE_TYPE}（ID：{PARENT_NODE_ID}），段落 [{SECTION_START}, {SECTION_END}]

**片段**：
{SEGMENT_WITH_SECTIONS}

---
决定是否细化，然后输出JSON。
""".strip()
