"""
Calculator Tool Module

Provides a safe calculator tool for the error detection agent to verify
arithmetic expressions found in reasoning chains.
"""

import ast
import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import AgentTool, AgentToolResult

log = logging.getLogger(__name__)


# Safe functions available for evaluation
SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "pow": pow,
    "pi": math.pi,
    "e": math.e,
}


def normalize_expression(expression: str) -> Optional[str]:
    """
    Normalize a mathematical expression for evaluation.

    Converts common mathematical notation to Python syntax:
    - × → *
    - ÷ → /
    - ^ → **
    - Various dash characters → -
    """
    if not expression:
        return None

    normalized = expression.strip()

    # Remove surrounding whitespace and common wrappers
    normalized = normalized.strip()

    # Convert mathematical symbols to Python operators
    normalized = normalized.replace("×", "*")
    normalized = normalized.replace("÷", "/")
    normalized = normalized.replace("^", "**")
    normalized = normalized.replace("−", "-")
    normalized = normalized.replace("–", "-")
    normalized = normalized.replace("—", "-")

    # Handle percentage
    normalized = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", normalized)

    # Normalize whitespace
    normalized = " ".join(normalized.split())

    return normalized if normalized else None


def _validate_ast(node: ast.AST) -> bool:
    """
    Validate an AST node to ensure it only contains safe operations.

    Prevents code injection by only allowing:
    - Numbers (int, float)
    - Basic arithmetic operations (+, -, *, /, **, //, %)
    - Unary operations (+, -)
    - Safe function calls (sqrt, abs, etc.)
    """
    if isinstance(node, ast.Expression):
        return _validate_ast(node.body)

    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float))

    # Python 3.7 compatibility
    if isinstance(node, ast.Num):
        return True

    if isinstance(node, ast.BinOp):
        allowed_ops = (
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Pow,
            ast.FloorDiv,
            ast.Mod,
        )
        if not isinstance(node.op, allowed_ops):
            return False
        return _validate_ast(node.left) and _validate_ast(node.right)

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            return False
        return _validate_ast(node.operand)

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            return all(_validate_ast(arg) for arg in node.args)
        return False

    if isinstance(node, ast.Name):
        # Allow constants like pi, e
        return node.id in SAFE_FUNCTIONS

    return False


def safe_eval(expression: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: The expression to evaluate

    Returns:
        Tuple of (result, error_message)
        - If successful: (float_result, None)
        - If failed: (None, error_description)
    """
    try:
        tree = ast.parse(expression, mode="eval")

        if not _validate_ast(tree.body):
            return None, "Expression contains disallowed operations"

        # Evaluate with restricted builtins
        result = eval(expression, {"__builtins__": {}}, SAFE_FUNCTIONS)

        return float(result), None

    except SyntaxError as e:
        return None, f"Syntax error: {e}"
    except ValueError as e:
        return None, f"Value error: {e}"
    except TypeError as e:
        return None, f"Type error: {e}"
    except ZeroDivisionError:
        return None, "Division by zero"
    except OverflowError:
        return None, "Result too large"
    except Exception as e:
        log.debug("Calculator eval failed for '%s': %s", expression, e)
        return None, f"Evaluation error: {e}"


def evaluate_expression_for_agent(
    expression: str,
    expected_result: Optional[str] = None,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Evaluate an expression and optionally compare with expected result.

    This is the main function used by the calculator tool.

    Args:
        expression: Mathematical expression to evaluate
        expected_result: Optional expected result to verify against
        tolerance: Tolerance for floating-point comparison

    Returns:
        Dictionary with:
        - expression: Original expression
        - normalized: Normalized expression
        - result: Computed result (or None if failed)
        - error: Error message (or None if successful)
        - expected: Expected result (if provided)
        - matches: Whether result matches expected (if expected provided)
        - difference: Absolute difference (if comparison made)
    """
    original = expression or ""

    # Handle expressions with = sign
    expr_part = original.strip()
    stated_result = expected_result

    if "=" in expr_part and not expected_result:
        parts = expr_part.split("=")
        expr_part = parts[0].strip()
        if len(parts) > 1:
            stated_result = parts[-1].strip()

    normalized = normalize_expression(expr_part)

    if not normalized:
        return {
            "expression": original,
            "normalized": None,
            "result": None,
            "error": "Empty or invalid expression",
            "expected": stated_result,
            "matches": None,
            "difference": None,
        }

    computed, error = safe_eval(normalized)

    result = {
        "expression": original,
        "normalized": normalized,
        "result": computed,
        "error": error,
        "expected": stated_result,
        "matches": None,
        "difference": None,
    }

    # Compare with expected if both available
    if computed is not None and stated_result:
        try:
            expected_float = float(stated_result)
            difference = abs(computed - expected_float)
            # Use relative tolerance for large numbers
            rel_tolerance = max(tolerance, abs(expected_float) * tolerance)
            matches = difference <= rel_tolerance

            result["matches"] = matches
            result["difference"] = difference
        except (ValueError, TypeError):
            # Expected is not a number
            result["error"] = (
                f"Cannot compare: expected '{stated_result}' is not numeric"
            )

    return result


class CalculatorTool(AgentTool):
    """
    Calculator tool for verifying arithmetic expressions.

    This tool can be called by the agent when it encounters
    mathematical calculations in the reasoning chain.
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluates a mathematical expression and returns the computed result. "
            "Can also verify if a stated result matches the computed value. "
            "Supports basic arithmetic (+, -, *, /, ^), parentheses, and common "
            "functions (sqrt, abs, sin, cos, tan, log, exp, etc.)."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The mathematical expression to evaluate. "
                        "Examples: '2 + 3 * 4', '(10 - 5) / 2', 'sqrt(16)', '2^10'"
                    ),
                },
                "expected_result": {
                    "type": "string",
                    "description": (
                        "Optional: The expected/stated result to verify. "
                        "If provided, will check if computed result matches."
                    ),
                },
            },
            "required": ["expression"],
        }

    def execute(self, **kwargs) -> AgentToolResult:
        """
        Execute the calculator tool.

        Args:
            expression: Mathematical expression to evaluate
            expected_result: Optional expected result to verify

        Returns:
            AgentToolResult with computation outcome
        """
        expression = kwargs.get("expression", "")
        expected_result = kwargs.get("expected_result")

        if not expression:
            return AgentToolResult(
                success=False,
                result=None,
                error="No expression provided",
            )

        eval_result = evaluate_expression_for_agent(
            expression=expression,
            expected_result=expected_result,
        )

        if eval_result.get("error"):
            return AgentToolResult(
                success=False,
                result=None,
                error=eval_result["error"],
                metadata=eval_result,
            )

        return AgentToolResult(
            success=True,
            result=eval_result["result"],
            metadata=eval_result,
        )
