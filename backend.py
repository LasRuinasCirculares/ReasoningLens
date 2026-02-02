"""
Backend server for chain-of‑thought reasoning and analysis.

This module implements a simple FastAPI service that exposes two endpoints:

  • `/api/reasoning` – Given a user question it queries a configured large language
    model to generate a chain‑of‑thought reasoning trace.

  • `/api/analyze` – Given a chain‑of‑thought trace, it queries a second model
    to break the trace into atomic reasoning units, flagging any erroneous
    steps and expressing the structure as a JSON graph (nodes and edges).

The models used for generation and analysis are configured via a YAML
configuration file (``config.yaml``). Each model definition includes
information about the provider (``openai`` or ``anthropic``), the model name
and any API keys or generation parameters. Environment variables may be
referenced using the ``${VAR_NAME}`` syntax in the config file; these
variables are automatically substituted at runtime.

This server is designed to be run locally for demonstration purposes. It
does not perform any network calls unless valid API keys are provided. If
the necessary libraries are not installed or an API call fails, the server
will raise a ``HTTPException`` with a descriptive message.
"""

from __future__ import annotations

import json
import os
import re
from bisect import bisect_right
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute environment variables in a YAML value.

    The config file may contain placeholders like ``${MY_ENV_VAR}``. This
    function will replace such placeholders with their corresponding
    environment variable values. If the variable is not set, it is replaced
    with an empty string.
    """
    if isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env_vars(v) for v in value]
    if isinstance(value, str):
        # Detect patterns like ${VAR} and replace them with os.environ values.
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replacer(match: re.Match) -> str:
            env_var = match.group(1)
            return os.environ.get(env_var, "")

        return pattern.sub(replacer, value)
    return value


def _load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load and process the YAML configuration file.

    This helper reads a YAML file, substitutes environment variables and
    returns a Python dictionary of the configuration. A missing file or
    parse error will raise a ``RuntimeError``.
    """
    if not os.path.exists(path):
        raise RuntimeError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    return _substitute_env_vars(raw_cfg)


class BaseModelWrapper:
    """Base class for LLM model wrappers.

    Subclasses implement the ``generate`` method to call the underlying
    provider. Each instance is configured via a dictionary containing
    provider‑specific fields (e.g. model name, API key, temperature).
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAIModel(BaseModelWrapper):
    """Wrapper for OpenAI's ChatCompletion API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import openai  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "openai python package is required for openai provider"
            ) from exc
        # openai>=1.0 client
        api_key = config.get("api_key")
        base_url = config.get("base_url")
        client_kwargs: Dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        try:
            self.client = openai.OpenAI(**client_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise OpenAI client: {exc}") from exc
        # Model and generation parameters
        self.model_name = config.get("model")
        self.temperature = float(config.get("temperature", 0.7))
        self.max_tokens = int(config.get("max_tokens", 2048))

    def generate(self, prompt: str) -> str:
        """Call the OpenAI ChatCompletion endpoint to generate text.

        This method uses a single user message containing the prompt and
        returns the assistant's reply as a string. Any exceptions raised by
        the API are propagated as HTTP errors to the client.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            # openai>=1.0 returns message.content as a list of text parts
            message = response.choices[0].message
            content = message.content
            if isinstance(content, list):
                # concatenate text parts
                text = "".join([part.text for part in content if hasattr(part, "text")])
            else:
                text = content or ""
            return text.strip()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API request failed: {exc}",
            )


class AnthropicModel(BaseModelWrapper):
    """Wrapper for Anthropic's API using the anthropic library."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import anthropic  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "anthropic python package is required for anthropic provider"
            ) from exc
        # Instantiate the Anthropic client
        api_key = config.get("api_key")
        if not api_key:
            raise RuntimeError(
                "Anthropic provider requires an 'api_key' in the configuration"
            )
        client_kwargs = {"api_key": api_key}
        base_url = config.get("base_url")
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = anthropic.Anthropic(**client_kwargs)
        self.model_name = config.get("model")
        self.temperature = float(config.get("temperature", 0.7))
        self.max_tokens = int(config.get("max_tokens", 2048))

    def generate(self, prompt: str) -> str:
        """Call the Anthropic API to generate text.

        Uses the Claude messages API. The prompt is sent as a single user
        message. Returns the assistant's reply as a string. Exceptions
        during the API call are propagated as HTTP errors.
        """
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            # Anthropic API returns a content list with text segments
            return "".join(
                part.text for part in message.content if hasattr(part, "text")
            ).strip()
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Anthropic API request failed: {exc}",
            )


def create_model(config: Dict[str, Any]) -> BaseModelWrapper:
    """Instantiate an LLM wrapper based on the provider specified in config."""
    provider = (config.get("provider") or "").lower()
    if provider == "openai":
        return OpenAIModel(config)
    if provider == "anthropic":
        return AnthropicModel(config)
    raise RuntimeError(f"Unsupported provider: {provider}")


def parse_analysis_result(raw_output: str) -> Dict[str, Any]:
    """Extract and parse the JSON graph from the analysis model output.

    Analysis models may prepend or append explanatory text around the JSON
    structure. This function searches for the first ``{`` and the last
    ``}`` in the response and attempts to parse the enclosed substring as
    JSON. It also tolerates common LLM formatting issues (markdown fences,
    trailing commas). If parsing fails, it raises an ``HTTPException``.
    """
    text = raw_output.strip()

    # Helper to attempt JSON parsing
    def _try_parse(candidate: str) -> Dict[str, Any]:
        return json.loads(candidate)

    # First, try to extract JSON inside ```json ... ``` fences if present
    fence_match = re.search(
        r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return _try_parse(candidate)
        except json.JSONDecodeError:
            # Fall back to more tolerant handling below
            pass

    # Fallback: find the substring that appears to be JSON
    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not json_match:
        raise HTTPException(
            status_code=500,
            detail="Analysis model output did not contain a JSON object",
        )
    json_str = json_match.group(0)

    # First strict parse
    try:
        data = _try_parse(json_str)
        return data
    except json.JSONDecodeError as exc:
        # Try a more tolerant parse by removing trailing commas before
        # closing braces/brackets, which LLMs sometimes introduce.
        cleaned = re.sub(r",(\s*[}\]])", r"\1", json_str)
        try:
            return _try_parse(cleaned)
        except json.JSONDecodeError as exc2:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse analysis output as JSON: {exc2}",
            )


def _normalise_graph(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure graph payload has the required shape for the frontend.

    Adds missing fields, coerces booleans and filters malformed edges so the
    UI can render without extra guards.
    """
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []

    normalised_nodes = []
    for idx, node in enumerate(nodes):
        substeps = []
        for step in node.get("substeps") or []:
            substeps.append(
                {
                    "text": step.get("text", ""),
                    "is_error": bool(step.get("is_error", False)),
                    "note": step.get("note"),
                }
            )

        normalised_nodes.append(
            {
                "id": node.get("id") or idx + 1,
                "label": node.get("label") or f"Step {idx + 1}",
                "content": node.get("content", ""),
                "hasError": bool(node.get("hasError", False)),
                "errorDescription": node.get("errorDescription"),
                "dependencies": node.get("dependencies") or [],
                "substeps": substeps,
            }
        )

    normalised_edges = []
    for edge in edges:
        if "from" in edge and "to" in edge:
            normalised_edges.append(
                {
                    "from": edge["from"],
                    "to": edge["to"],
                    "type": edge.get("type", "normal"),
                }
            )

    return {"nodes": normalised_nodes, "edges": normalised_edges}


def _build_line_offsets(text: str) -> List[int]:
    """Return a list of starting offsets for each line in ``text``."""
    offsets = [0]
    for match in re.finditer(r"\n", text):
        offsets.append(match.end())
    return offsets


def _offset_to_line(offsets: List[int], idx: int) -> int:
    """Convert a character offset to a 1-based line number using ``offsets``."""
    position = max(idx, 0)
    return max(bisect_right(offsets, position), 1)


def _attach_reasoning_spans(nodes: List[Dict[str, Any]], reasoning_chain: str) -> None:
    """Annotate nodes with approximate text ranges within the reasoning chain."""
    if not reasoning_chain or not nodes:
        return

    lowered_chain = reasoning_chain.lower()
    line_offsets = _build_line_offsets(reasoning_chain)
    search_start = 0

    for node in nodes:
        content = (node.get("content") or "").strip()
        if not content:
            continue

        snippet = content.lower()
        start_idx = lowered_chain.find(snippet, search_start)
        if start_idx == -1:
            start_idx = lowered_chain.find(snippet)
        if start_idx == -1:
            continue

        end_idx = start_idx + len(snippet)
        node["textRange"] = {
            "start": start_idx,
            "end": end_idx,
            "startLine": _offset_to_line(line_offsets, start_idx),
            "endLine": _offset_to_line(line_offsets, end_idx - 1),
        }
        search_start = end_idx


# Load configuration and instantiate models at import time
CONFIG_PATH = os.environ.get("CONFIG_PATH") or os.path.join(
    os.path.dirname(__file__), "config.yaml"
)
CONFIG = _load_config(CONFIG_PATH)
MODELS_CFG = CONFIG.get("models", {})
SERVER_CFG = CONFIG.get("server", {})
BASE_DIR = os.path.dirname(__file__)
LOG_DIR = os.path.join(BASE_DIR, SERVER_CFG.get("log_dir", "logs"))

DEFAULT_DECOMPOSITION_PROMPT = """
You are a reasoning-chain analyst. Decompose the long chain-of-thought into ordered atomic units based on key transitions such as planning, reasoning, backtracking, self-validation, and reflection. Preserve branching and backtracking links.
""".strip()

DEFAULT_ERROR_PROMPT = """
For each atomic unit, run error checks using these categories (set hasError=true and explain in errorDescription when applicable):
- Ruminate: stuck in loops or repetitive thinking.
- Ineffective Reflection: reflection is flawed or unhelpful.
- Incoherent Content: disjointed or non-continuous text / poor instruction following.
- Format Error: output violates expected format or is incomplete.
- Knowledge Errors: incorrect facts/definitions/theorems.
- Logical Error: wrong strategy, invalid premises, incoherent or contradictory reasoning.
- Formal Manipulation Error: rule violations in math/code/logic/physics computations.
- Faithfulness Issue: conflicts with user-provided info; reasoning not aligned.
- Safety Issue: harmful/unsafe/discriminatory content or missing mitigation.

Output JSON only:
{
  "nodes": [
    {
      "id": 1,
      "label": "Short label",
      "content": "Full content of this atomic unit",
      "hasError": false,
      "errorDescription": null,
      "dependencies": [ids of prerequisites],
      "substeps": [
        {"text": "Substep 1", "is_error": false, "note": "optional"},
        {"text": "Substep 2", "is_error": true, "note": "why it is wrong"}
      ]
    }
  ],
  "edges": [
    {"from": 1, "to": 2, "type": "normal"},
    {"from": 3, "to": 2, "type": "backtrack"}
  ]
}
""".strip()

try:
    reasoning_model_cfg = MODELS_CFG.get("reasoning_model", {})
    analysis_model_cfg = MODELS_CFG.get("analysis_model", {})
    reasoning_model = create_model(reasoning_model_cfg)
    analysis_model = create_model(analysis_model_cfg)
    # The analysis prompt template can be customised in config
    DECOMP_PROMPT = (
        analysis_model_cfg.get("decomposition_prompt_template")
        or DEFAULT_DECOMPOSITION_PROMPT
    )
    ERROR_PROMPT = (
        analysis_model_cfg.get("error_prompt_template") or DEFAULT_ERROR_PROMPT
    )
except Exception as exc:
    # If model instantiation fails, raise a runtime error so the app won't start
    raise RuntimeError(f"Failed to initialise models: {exc}")


app = FastAPI(title="Reasoning Analysis API")

if SERVER_CFG.get("cors_origins"):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=SERVER_CFG["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class ReasoningRequest(BaseModel):
    question: str


class AnalysisRequest(BaseModel):
    reasoning_chain: str


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append a single JSON record as one line to ``path``.

    Used to persist model interactions for later inspection/evaluation.
    Failures are swallowed so logging never breaks the main flow.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Best-effort logging; ignore any filesystem errors
        return


@app.post("/api/reasoning")
async def generate_reasoning(req: ReasoningRequest) -> Dict[str, Any]:
    """Generate a chain‑of‑thought reasoning trace for a given question."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty")
    # Call the reasoning model; wrap any errors into HTTP exceptions
    reasoning = reasoning_model.generate(question)
    # Persist question and model reply for later evaluation
    log_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": "reasoning",
        "question": question,
        "reasoning_chain": reasoning,
    }
    _append_jsonl(os.path.join(LOG_DIR, "reasoning_log.jsonl"), log_record)
    return {"reasoning_chain": reasoning}


@app.post("/api/analyze")
async def analyze_reasoning(req: AnalysisRequest) -> Dict[str, Any]:
    """Analyse a chain‑of‑thought and return a structured reasoning graph."""
    reasoning_chain = req.reasoning_chain.strip()
    if not reasoning_chain:
        raise HTTPException(status_code=400, detail="Reasoning chain must not be empty")
    # Construct the prompt combining decomposition and error-check templates.
    prompt_parts = [
        DECOMP_PROMPT.strip(),
        ERROR_PROMPT.strip(),
        "=== REASONING CHAIN ===",
        reasoning_chain,
        "=== END OF CHAIN ===",
        "Generate JSON per the schema.",
    ]
    prompt = "\n\n".join([p for p in prompt_parts if p])
    # Call the analysis model
    raw_analysis = analysis_model.generate(prompt)
    # Parse and normalise the structured result
    parsed = parse_analysis_result(raw_analysis)
    result = _normalise_graph(parsed)
    _attach_reasoning_spans(result["nodes"], reasoning_chain)
    return result


@app.get("/api/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=int(SERVER_CFG.get("port", 8000)),
        reload=True,
    )
