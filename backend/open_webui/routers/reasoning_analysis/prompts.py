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
    "You are a reasoning structure annotator. "
    "Your task is to analyze a chain-of-thought reasoning process and convert it into a coarse-grained graph. "
    "The reasoning text has already been split into numbered sections, and you must specify which sections each node covers. "
    "Use exactly FIVE node types: 'problem_decomposition', 'reasoning_step', 'check', 'intermediate_answer', 'final_answer'. "
    "Use exactly THREE edge types: 'reasoning', 'check', and 'backtracking'. "
    "Return JSON only. Do not output markdown, commentary, or code fences."
)

LAYER1_USER_PROMPT_TEMPLATE_EN = """
Your goal is to read the sectioned reasoning trace, identify its high-level reasoning structure, and output a coarse-grained graph in JSON format. Focus on the actual organization of the reasoning rather than surface repetition or line-by-line details.

The input has already been split into numbered sections `[Section 1]`, `[Section 2]`, etc.

## Node Type Definitions

Use exactly these five node types:

1. `problem_decomposition`
   - Initial understanding of the problem
   - Breaking the task into subgoals
   - Setting up a plan or framing the target

2. `reasoning_step`
   - A main reasoning move
   - Derivation, case analysis, search, comparison, weighing factors, or other substantive step that advances the solution

3. `check`
   - A distinct verification action
   - Rechecking a result, substituting back, validating a candidate, or testing whether a previous step is correct
   - Metadata must include `validates_node`

4. `intermediate_answer`
   - An explicit candidate answer to the original question that appears before the reasoning has finished
   - Use this only when the trace states a concrete provisional answer, answer set, or competing candidate answer to the main question and then continues to check it, compare it, revise it, or reason further
   - Do NOT use this for temporary calculations, local scores, partial summaries, feasibility checks, or transition sentences
   - Metadata must include `from_node` and `answer_value`

5. `final_answer`
   - The final concrete answer to the original question
   - Use this for the terminal answer statement, not for a generic summary
   - There must be exactly one `final_answer` node
   - Metadata must include `from_node` and `answer_value`

## Edge Type Definitions

Use exactly these three edge types:

1. `reasoning`
   - Normal forward reasoning flow from one node to the next.
   - Use this for ordinary progression where one step leads to another.

   Example:
   `[Problem setup] --reasoning--> [Main derivation] --reasoning--> [Intermediate answer]`

2. `check`
   - A verification edge FROM the checking node TO the checked node.
   - Use this when a node verifies, substitutes back, validates, or tests a
     previous reasoning step or answer.
   - The edge direction must be:
     `[Check node] --check--> [Checked node]`

   Example:
   ```
   [Reasoning] --reasoning--> [Answer: x = 5] --reasoning--> [Final confirmation]
                                ^
                                |
                    [Check: substitute x = 5 back]
   ```
   In this example, the check edge is:
   `[Check: substitute x = 5 back] --check--> [Answer: x = 5]`

3. `backtracking`
   - Backtracking is a failure-triggered reasoning behavior invoked when the
     current strategy cannot produce a feasible solution.
   - It prunes the failed branch and resumes search from a prior decision point
     to explore alternatives.
   - Use this when the trace abandons a failed approach, interpretation, or
     intermediate result and opens a new alternative branch.
   - Prefer attaching the `backtracking` edge FROM the failed node itself
     (usually a wrong `reasoning_step` or wrong `intermediate_answer`) TO the
     new alternative `reasoning_step`.
   - The earlier pivot node should also connect to the alternative branch with a
     normal `reasoning` edge, so the two paths are represented as parallel
     branches rather than a single linear rewrite.

   Example:
   ```
   [Pivot] --reasoning--> [Approach A] --reasoning--> [Wrong result]
      |
      |--reasoning----------------------------------> [Approach B] --reasoning--> [Correct answer]

   and also:

   [Wrong result] --backtracking--> [Approach B]
   ```

## Section Mapping Requirements

Each node must cover a contiguous section range using:
- `section_start`
- `section_end`

These ranges are inclusive and 1-indexed.

Hard constraints:
- Cover ALL sections from `1` to `{SECTION_COUNT}`
- NO gaps
- NO overlap
- Each section can appear in exactly ONE node
- Section ranges must be in ascending order
- Each node should represent one complete logical unit, not a fragmented micro-step

## Structural Requirements

- Build a coarse-grained graph, not a line-by-line decomposition
- Typical traces should usually need around 3-8 nodes, though longer traces may require more
- Every node must participate in the graph
- Use `check` only when there is a distinct validation action
- Use `intermediate_answer` only for explicit candidate answers to the original question
- Use `backtracking` only for genuine failed-path switching
- The graph should reflect the high-level reasoning organization, not noisy repetition

## Evidence Requirement

Each node must include:
- `evidence_quote`: a short verbatim quote from the covered sections supporting the node

Keep the quote short and specific.

## Output Format

Return ONLY valid JSON using this top-level schema:

{{
  "summary": "Brief overview of the reasoning structure",
  "nodes": [
    {{
      "id": "node1",
      "type": "problem_decomposition",
      "label": "Short label",
      "description": "What this node does",
      "section_start": 1,
      "section_end": 3,
      "evidence_quote": "Short supporting quote",
      "metadata": {{}}
    }},
    {{
      "id": "node2",
      "type": "intermediate_answer",
      "label": "Candidate answer",
      "description": "The model states a provisional answer before further reasoning.",
      "section_start": 4,
      "section_end": 4,
      "evidence_quote": "a candidate answer is ...",
      "metadata": {{"from_node": "node1", "answer_value": "..." }}
    }},
    {{
      "id": "node3",
      "type": "final_answer",
      "label": "Final answer",
      "description": "The terminal answer to the question.",
      "section_start": 5,
      "section_end": 5,
      "evidence_quote": "therefore the answer is ...",
      "metadata": {{"from_node": "node2", "answer_value": "..." }}
    }}
  ],
  "edges": [
    {{
      "from": "node1",
      "to": "node2",
      "type": "reasoning",
      "label": "optional short edge label"
    }}
  ]
}}

## Final Checks Before Output

- Only the 5 allowed node types are used
- Only the 3 allowed edge types are used
- All sections `1..{SECTION_COUNT}` are covered exactly once
- There is no section overlap and no missing section
- Every node has a short `evidence_quote`
- Every `intermediate_answer` has `from_node` and `answer_value`
- Exactly one `final_answer` exists and it has `from_node` and `answer_value`
- The graph reflects high-level reasoning structure rather than repetitive surface text

Now analyze the following input:

Question:
{QUESTION}

Total Sections:
{SECTION_COUNT}

Reasoning:
{REASONING_WITH_SECTIONS}
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
    "你是一位推理结构标注者。"
    "你的任务是分析思维链推理过程，并将其转换为一个粗粒度图结构。"
    "推理文本已经被预先切分成编号段落，你必须为每个节点指定覆盖的段落范围。"
    "严格使用5种节点类型：'problem_decomposition'(问题分解), 'reasoning_step'(推理步骤), 'check'(验证节点), 'intermediate_answer'(中间答案), 'final_answer'(最终答案)。"
    "严格使用3种边类型：'reasoning'、'check'、'backtracking'。"
    "只返回JSON，不要输出markdown、解释或代码块。"
)

LAYER1_USER_PROMPT_TEMPLATE_ZH = """
你的目标是阅读已经按段编号的推理文本，识别其高层推理结构，并以 JSON 格式输出一个粗粒度图。重点关注真实的推理组织方式，而不是表面重复或逐句切分。

输入已经被切分成编号段落，如 `[Section 1]`、`[Section 2]` 等。

## 节点类型定义

严格使用以下五种节点类型：

1. `problem_decomposition`
   - 对问题的初始理解
   - 将任务拆成子目标
   - 建立计划或分析框架

2. `reasoning_step`
   - 一个主要推理动作
   - 推导、分类讨论、搜索、比较、权衡因素，或其他实质性推进求解的步骤

3. `check`
   - 一个独立的验证动作
   - 回代检验、验证候选答案、确认前一步是否成立
   - metadata 必须包含 `validates_node`

4. `intermediate_answer`
   - 在推理尚未结束前，针对原问题给出的明确候选答案
   - 只有当文本明确说出一个针对主问题的暂定答案、候选答案或竞争答案，且后续还会继续验证、比较、修正或替换它时，才使用这个类型
   - 不要将临时计算、局部打分、阶段性总结、可行性检查或过渡句标成这个类型
   - metadata 必须包含 `from_node` 和 `answer_value`

5. `final_answer`
   - 针对原问题的最终具体答案
   - 这个类型用于最终答案陈述，而不是泛化总结
   - 必须且只能有一个 `final_answer` 节点
   - metadata 必须包含 `from_node` 和 `answer_value`

## 边类型定义

严格使用以下三种边类型：

1. `reasoning`
   - 正常的前向推理流程

2. `check`
   - 从验证节点指向被验证节点的验证边
   - 被验证节点可以是 `reasoning_step`、`intermediate_answer` 或 `final_answer`

3. `backtracking`
   - 由于前一路径失败或被放弃而切换到替代路径
   - 只有当文本明确否定、放弃或认定某条路径失败，然后开启另一条真正替代路径时，才使用这个类型
   - 不要把简单补充、改写或沿同一路径继续展开标成 `backtracking`

## 段落映射要求

每个节点都必须包含一个连续的段落范围：
- `section_start`
- `section_end`

范围是闭区间，并且从 1 开始编号。

硬约束：
- 必须覆盖 `1` 到 `{SECTION_COUNT}` 的所有段落
- 不能有缺口
- 不能有重叠
- 每个段落只能属于一个节点
- 段落范围必须按顺序递增
- 每个节点应代表一个完整逻辑单元，而不是碎片化的小步骤

## 结构要求

- 构建的是粗粒度图，不是逐句分解
- 常见推理通常只需要 3-8 个节点，较长推理可以更多
- 每个节点都必须参与图结构
- 只有出现明确验证动作时才使用 `check`
- 只有出现针对原问题的明确候选答案时才使用 `intermediate_answer`
- 只有出现真正的失败后换路时才使用 `backtracking`
- 图应反映高层推理组织，而不是重复的表面文本

## 证据要求

每个节点都必须包含：
- `evidence_quote`：来自该节点覆盖段落的简短原文引语，用于支持该节点标注

引语要短且具体。

## 输出格式

只返回合法 JSON，顶层结构如下：

{{
  "summary": "对推理结构的简短概述",
  "nodes": [
    {{
      "id": "node1",
      "type": "problem_decomposition",
      "label": "简短标签",
      "description": "该节点的作用",
      "section_start": 1,
      "section_end": 3,
      "evidence_quote": "简短证据引文",
      "metadata": {{}}
    }},
    {{
      "id": "node2",
      "type": "intermediate_answer",
      "label": "候选答案",
      "description": "在继续推理前先给出一个暂定答案。",
      "section_start": 4,
      "section_end": 4,
      "evidence_quote": "一个候选答案是……",
      "metadata": {{"from_node": "node1", "answer_value": "..." }}
    }},
    {{
      "id": "node3",
      "type": "final_answer",
      "label": "最终答案",
      "description": "对问题的最终回答。",
      "section_start": 5,
      "section_end": 5,
      "evidence_quote": "因此答案是……",
      "metadata": {{"from_node": "node2", "answer_value": "..." }}
    }}
  ],
  "edges": [
    {{
      "from": "node1",
      "to": "node2",
      "type": "reasoning",
      "label": "可选的简短边标签"
    }}
  ]
}}

## 输出前最终检查

- 只使用 5 种允许的节点类型
- 只使用 3 种允许的边类型
- 所有 `1..{SECTION_COUNT}` 段落都被且仅被覆盖一次
- 没有段落重叠，也没有缺失段落
- 每个节点都有简短的 `evidence_quote`
- 每个 `intermediate_answer` 都有 `from_node` 和 `answer_value`
- 必须存在且仅存在一个 `final_answer`，并且它有 `from_node` 和 `answer_value`
- 图反映的是高层推理结构，而不是重复的表面文本

现在分析以下输入：

问题：
{QUESTION}

总段落数：
{SECTION_COUNT}

推理过程：
{REASONING_WITH_SECTIONS}
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
