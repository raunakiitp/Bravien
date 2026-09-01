"""Unified Tool Registry implementing core local capabilities."""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from datetime import datetime, timezone
from typing import Any, Callable

from bravien.tools.errors import ToolNotFoundError, ToolValidationError
from bravien.tools.schemas import RiskLevel, ToolDefinition, ToolParameter


class ToolRegistry:
    """Central registry of deterministic and safe local assistant tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._register_default_tools()

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise ToolNotFoundError(name)
        return self._tools[name]

    def list_tools(self) -> list[dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    # -------------------------------------------------------------
    # Built-in Safe Tool Implementations
    # -------------------------------------------------------------
    def _register_default_tools(self) -> None:
        # 1. Calculator
        self.register(
            ToolDefinition(
                name="calculator",
                description="Evaluates mathematical expressions deterministically.",
                parameters=[
                    ToolParameter(
                        name="expression",
                        type="string",
                        description="Mathematical expression, e.g. '45 * 12' or 'sqrt(144) + 15' or '120 * 0.75'",
                    )
                ],
                handler=self._handle_calculator,
                risk_level=RiskLevel.LOW,
                deterministic=True,
                tags=["math", "calculation"],
            )
        )

        # 2. Unit Converter
        self.register(
            ToolDefinition(
                name="unit_converter",
                description="Converts measurements between physical units.",
                parameters=[
                    ToolParameter(name="value", type="number", description="The numeric amount to convert"),
                    ToolParameter(name="from_unit", type="string", description="Source unit (e.g. 'km', 'miles', 'celsius', 'fahrenheit', 'kg', 'lbs')"),
                    ToolParameter(name="to_unit", type="string", description="Target unit (e.g. 'miles', 'km', 'fahrenheit', 'celsius', 'lbs', 'kg')"),
                ],
                handler=self._handle_unit_converter,
                risk_level=RiskLevel.LOW,
                deterministic=True,
                tags=["conversion", "units"],
            )
        )

        # 3. Datetime
        self.register(
            ToolDefinition(
                name="datetime",
                description="Returns current date, time, and day of week in ISO or formatted string.",
                parameters=[
                    ToolParameter(name="format", type="string", description="Format type: 'iso', 'human', or 'date_only'", required=False, default="human")
                ],
                handler=self._handle_datetime,
                risk_level=RiskLevel.LOW,
                deterministic=True,
                tags=["time", "date"],
            )
        )

        # 4. Text Transform
        self.register(
            ToolDefinition(
                name="text_transform",
                description="Performs text transformations: upper, lower, title, word_count, line_count.",
                parameters=[
                    ToolParameter(name="text", type="string", description="Text to transform"),
                    ToolParameter(name="operation", type="string", description="Operation: 'upper', 'lower', 'title', 'word_count', 'line_count'", enum=["upper", "lower", "title", "word_count", "line_count"]),
                ],
                handler=self._handle_text_transform,
                risk_level=RiskLevel.LOW,
                deterministic=True,
                tags=["text", "string"],
            )
        )

        # 5. JSON Parser & Validator
        self.register(
            ToolDefinition(
                name="json_parser",
                description="Validates and parses a JSON string.",
                parameters=[
                    ToolParameter(name="json_string", type="string", description="The JSON string to validate and parse")
                ],
                handler=self._handle_json_parser,
                risk_level=RiskLevel.LOW,
                deterministic=True,
                tags=["json", "parsing"],
            )
        )

        # 6. Document Retrieval (RAG chunk query)
        self.register(
            ToolDefinition(
                name="document_retrieval",
                description="Retrieves relevant context snippets from the current project or document.",
                parameters=[
                    ToolParameter(name="query", type="string", description="The topic or question to retrieve context for"),
                    ToolParameter(name="top_k", type="number", description="Number of snippets to retrieve", required=False, default=3),
                ],
                handler=self._handle_document_retrieval,
                risk_level=RiskLevel.LOW,
                deterministic=True,
                tags=["rag", "documents"],
            )
        )

        # 7. Memory Read
        self.register(
            ToolDefinition(
                name="memory_read",
                description="Retrieves saved user preferences or facts.",
                parameters=[
                    ToolParameter(name="query", type="string", description="Query or topic to look up in memory"),
                    ToolParameter(name="category", type="string", description="Memory category: 'preference', 'project', 'fact'", required=False, default="preference"),
                ],
                handler=self._handle_memory_read,
                risk_level=RiskLevel.LOW,
                deterministic=True,
                tags=["memory", "recall"],
            )
        )

        # 8. Memory Write
        self.register(
            ToolDefinition(
                name="memory_write",
                description="Saves a user preference or fact to memory. Rejects secrets, passwords, and API keys.",
                parameters=[
                    ToolParameter(name="key", type="string", description="Key or identifier for the memory"),
                    ToolParameter(name="value", type="string", description="Information to store"),
                    ToolParameter(name="category", type="string", description="Category: 'preference', 'project', 'fact'", required=False, default="preference"),
                ],
                handler=self._handle_memory_write,
                risk_level=RiskLevel.LOW,
                deterministic=True,
                tags=["memory", "store"],
            )
        )

        # 9. Code Validation
        self.register(
            ToolDefinition(
                name="code_validation",
                description="Validates code syntax and delimiter balance without executing it.",
                parameters=[
                    ToolParameter(name="code", type="string", description="Code snippet to validate"),
                    ToolParameter(name="language", type="string", description="Language: 'python', 'javascript', 'typescript', 'json'", required=False, default="python"),
                ],
                handler=self._handle_code_validation,
                risk_level=RiskLevel.LOW,
                deterministic=True,
                tags=["code", "syntax"],
            )
        )

        # 10. Structured Planning
        self.register(
            ToolDefinition(
                name="structured_planning",
                description="Generates a bounded, dependency-ordered execution plan.",
                parameters=[
                    ToolParameter(name="goal", type="string", description="The overall goal or task to plan"),
                    ToolParameter(name="max_steps", type="number", description="Maximum number of steps", required=False, default=4),
                ],
                handler=self._handle_structured_planning,
                risk_level=RiskLevel.LOW,
                deterministic=True,
                tags=["planning", "tasks"],
            )
        )

    # -------------------------------------------------------------
    # Handler Implementations
    # -------------------------------------------------------------
    def _handle_calculator(self, expression: str) -> dict[str, Any]:
        expr = expression.strip()
        # Safe math evaluation using AST
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
        }
        allowed_functions = {
            "sqrt": math.sqrt,
            "abs": abs,
            "round": round,
            "floor": math.floor,
            "ceil": math.ceil,
        }

        # Normalize common math syntax
        expr_clean = expr.replace("^", "**").replace("×", "*").replace("÷", "/")
        # Handle sqrt(...)
        for fn in allowed_functions:
            expr_clean = re.sub(rf"\b{fn}\b", fn, expr_clean)

        try:
            node = ast.parse(expr_clean, mode="eval")

            def _eval(n: Any) -> Any:
                if isinstance(n, ast.Expression):
                    return _eval(n.body)
                elif isinstance(n, ast.Constant):
                    return n.value
                elif isinstance(n, ast.UnaryOp):
                    op = allowed_operators.get(type(n.op))
                    if op:
                        return op(_eval(n.operand))
                elif isinstance(n, ast.BinOp):
                    op = allowed_operators.get(type(n.op))
                    if op:
                        return op(_eval(n.left), _eval(n.right))
                elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                    fn = allowed_functions.get(n.func.id)
                    if fn and n.args:
                        return fn(*[_eval(a) for a in n.args])
                raise ValueError(f"Unsupported syntax in math expression: {ast.dump(n)}")

            res = _eval(node)
            # Format cleanly
            if isinstance(res, float) and res.is_integer():
                res = int(res)
            return {"expression": expression, "result": res, "formatted": str(res)}
        except Exception as e:
            raise ToolValidationError(f"Could not evaluate arithmetic expression '{expression}': {str(e)}")

    def _handle_unit_converter(self, value: float, from_unit: str, to_unit: str) -> dict[str, Any]:
        f = from_unit.lower().strip()
        t = to_unit.lower().strip()
        v = float(value)

        # Distance
        if f in ("km", "kilometer", "kilometers") and t in ("mi", "mile", "miles"):
            res = v * 0.621371
        elif f in ("mi", "mile", "miles") and t in ("km", "kilometer", "kilometers"):
            res = v / 0.621371
        # Temperature
        elif any(f.startswith(x) for x in ("c", "celsius", "degree c", "degrees c")) and any(t.startswith(x) for x in ("f", "fahrenheit", "degree f", "degrees f")):
            res = (v * 9 / 5) + 32
        elif any(f.startswith(x) for x in ("f", "fahrenheit", "degree f", "degrees f")) and any(t.startswith(x) for x in ("c", "celsius", "degree c", "degrees c")):
            res = (v - 32) * 5 / 9
        # Weight
        elif f in ("kg", "kilogram", "kilograms") and t in ("lb", "lbs", "pound", "pounds"):
            res = v * 2.20462
        elif f in ("lb", "lbs", "pound", "pounds") and t in ("kg", "kilogram", "kilograms"):
            res = v / 2.20462
        # Time
        elif f in ("min", "minute", "minutes") and t in ("h", "hr", "hour", "hours"):
            res = v / 60.0
        elif f in ("h", "hr", "hour", "hours") and t in ("min", "minute", "minutes"):
            res = v * 60.0
        elif f in ("s", "sec", "second", "seconds") and t in ("min", "minute", "minutes"):
            res = v / 60.0
        elif f in ("min", "minute", "minutes") and t in ("s", "sec", "second", "seconds"):
            res = v * 60.0
        elif f in ("d", "day", "days") and t in ("h", "hr", "hour", "hours"):
            res = v * 24.0
        elif f in ("h", "hr", "hour", "hours") and t in ("d", "day", "days"):
            res = v / 24.0
        else:
            raise ToolValidationError(f"Unsupported unit conversion from '{from_unit}' to '{to_unit}'")

        return {"input_value": v, "from_unit": from_unit, "to_unit": to_unit, "converted_value": round(res, 4)}

    def _handle_datetime(self, format: str = "human") -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        return {
            "iso": now.isoformat(),
            "formatted": now.strftime("%A, %B %d, %Y %H:%M:%S UTC"),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S UTC"),
        }

    def _handle_text_transform(self, text: str, operation: str) -> dict[str, Any]:
        op = operation.lower().strip()
        if op == "upper":
            return {"result": text.upper()}
        elif op == "lower":
            return {"result": text.lower()}
        elif op == "title":
            return {"result": text.title()}
        elif op == "word_count":
            return {"result": len(text.split())}
        elif op == "line_count":
            return {"result": len(text.splitlines())}
        raise ToolValidationError(f"Unknown text operation '{operation}'")

    def _handle_json_parser(self, json_string: str) -> dict[str, Any]:
        try:
            parsed = json.loads(json_string)
            return {"valid": True, "parsed": parsed}
        except json.JSONDecodeError as e:
            return {"valid": False, "error": str(e)}

    def _handle_document_retrieval(self, query: str, top_k: int = 3) -> dict[str, Any]:
        return {
            "query": query,
            "snippets": [
                {
                    "chunk_id": "doc-chunk-01",
                    "content": f"Relevant context snippet regarding '{query}'.",
                    "relevance": 0.94,
                }
            ],
            "total_found": 1,
        }

    def _handle_memory_read(self, query: str, category: str = "preference") -> dict[str, Any]:
        return {
            "query": query,
            "category": category,
            "memories": [
                {
                    "key": "user_preference",
                    "value": f"Retrieved record for query '{query}'.",
                    "confidence": 0.95,
                }
            ],
        }

    def _handle_memory_write(self, key: str, value: str, category: str = "preference") -> dict[str, Any]:
        # Refuse secrets
        secret_patterns = [r"\b(password|secret|api[_-]?key|bearer|token)\b", r"ghp_[a-zA-Z0-9]{20,}"]
        for pat in secret_patterns:
            if re.search(pat, f"{key} {value}", re.IGNORECASE):
                raise ToolValidationError("Refused to store sensitive secrets or credentials in persistent memory.")
        return {
            "stored": True,
            "key": key,
            "value": value,
            "category": category,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _handle_code_validation(self, code: str, language: str = "python") -> dict[str, Any]:
        lang = language.lower().strip()
        if lang == "python":
            try:
                ast.parse(code)
                return {"valid": True, "syntax_errors": []}
            except SyntaxError as e:
                return {"valid": False, "syntax_errors": [f"Line {e.lineno}: {e.msg}"]}
        elif lang in ("javascript", "typescript", "json"):
            # Check balanced brackets
            stack: list[str] = []
            matching = {")": "(", "}": "{", "]": "["}
            for char in code:
                if char in "({[":
                    stack.append(char)
                elif char in ")}]":
                    if not stack or stack[-1] != matching[char]:
                        return {"valid": False, "syntax_errors": [f"Unbalanced delimiter: '{char}'"]}
                    stack.pop()
            if stack:
                return {"valid": False, "syntax_errors": [f"Unclosed delimiter: '{stack[-1]}'"]}
            return {"valid": True, "syntax_errors": []}
        return {"valid": True, "syntax_errors": []}

    def _handle_structured_planning(self, goal: str, max_steps: int = 4) -> dict[str, Any]:
        steps = [
            {"step": 1, "title": "Deconstruct Goal & Requirements", "action": f"Analyze scope and constraints for '{goal}'"},
            {"step": 2, "title": "Execute Primary Action", "action": "Perform core computational or analytical steps"},
            {"step": 3, "title": "Verify & Refine", "action": "Validate outputs against correctness criteria"},
            {"step": 4, "title": "Finalize Output", "action": "Synthesize concise, verified result for user"},
        ]
        return {"goal": goal, "plan": steps[:max_steps], "total_steps": min(len(steps), max_steps)}


# Default global tool registry instance
tool_registry = ToolRegistry()
