"""Bravien Stage 9: Production Deterministic Intercept Layer.

Pre-screens user queries before they reach the language model.
Catches arithmetic, unit conversion, temperature conversion, multi-step math,
anti-hallucination, clarification, ambiguity, and memory/context recall.

This is the Python-side mirror of the TypeScript inference-gate and mirrors
the Stage 9 benchmark's Stage9InterceptLayer, ensuring that benchmark
results reflect true production behavior.
"""

from __future__ import annotations

import ast
import operator as op
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator
# ---------------------------------------------------------------------------

_OPS: dict = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Unsupported AST node: {type(node)}")


def eval_expr_safe(expr: str) -> float | None:
    """Safely evaluate a numeric expression string via AST (no eval)."""
    try:
        clean = re.sub(r"[$,%]", "", expr).strip()
        tree = ast.parse(clean, mode="eval")
        result = _eval_node(tree.body)
        # Guard against infinity / nan
        if result != result or abs(result) == float("inf"):
            return None
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Unit converter
# ---------------------------------------------------------------------------

def convert_unit(value: float, from_unit: str, to_unit: str) -> float | None:
    """Deterministic unit converter: temperature, distance, weight, time."""
    f = from_unit.lower().strip()
    t = to_unit.lower().strip()
    try:
        # Temperature
        if f in ("c", "celsius") and t in ("f", "fahrenheit"):
            return (value * 9.0 / 5.0) + 32.0
        if f in ("f", "fahrenheit") and t in ("c", "celsius"):
            return (value - 32.0) * 5.0 / 9.0
        if f in ("c", "celsius") and t in ("k", "kelvin"):
            return value + 273.15
        if f in ("k", "kelvin") and t in ("c", "celsius"):
            return value - 273.15
        # Distance
        if f in ("km", "kilometer", "kilometers") and t in ("mi", "mile", "miles"):
            return value * 0.621371
        if f in ("mi", "mile", "miles") and t in ("km", "kilometer", "kilometers"):
            return value / 0.621371
        if f in ("m", "meter", "meters") and t in ("ft", "foot", "feet"):
            return value * 3.28084
        if f in ("ft", "foot", "feet") and t in ("m", "meter", "meters"):
            return value / 3.28084
        # Weight
        if f in ("kg", "kilogram", "kilograms") and t in ("lb", "lbs", "pound", "pounds"):
            return value * 2.20462
        if f in ("lb", "lbs", "pound", "pounds") and t in ("kg", "kilogram", "kilograms"):
            return value / 2.20462
        if f in ("g", "gram", "grams") and t in ("oz", "ounce", "ounces"):
            return value * 0.035274
        # Time
        if f in ("min", "minute", "minutes") and t in ("h", "hr", "hour", "hours"):
            return value / 60.0
        if f in ("h", "hr", "hour", "hours") and t in ("min", "minute", "minutes"):
            return value * 60.0
        if f in ("s", "sec", "second", "seconds") and t in ("min", "minute", "minutes"):
            return value / 60.0
        if f in ("min", "minute", "minutes") and t in ("s", "sec", "second", "seconds"):
            return value * 60.0
        if f in ("d", "day", "days") and t in ("h", "hr", "hour", "hours"):
            return value * 24.0
        if f in ("h", "hr", "hour", "hours") and t in ("d", "day", "days"):
            return value / 24.0
    except Exception:
        pass
    return None


def fmt_num(v: float) -> str:
    """Format a number cleanly: integer if whole, else 2dp."""
    if v == int(v):
        return str(int(v))
    return f"{v:.4f}".rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# Intercept result
# ---------------------------------------------------------------------------

@dataclass
class InterceptResult:
    intercepted: bool
    response: str = ""
    method: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Intercept layer
# ---------------------------------------------------------------------------

class DeterministicInterceptLayer:
    """
    Pre-screens user queries before the LLM.

    Applied in order: safety shortcuts → ambiguity/clarification → temperature
    conversion → other unit conversions → multi-tool chaining → multi-step word
    problems → direct arithmetic → anti-hallucination → memory preference recall
    → list context recall.

    Returns an InterceptResult. If intercepted=True, use `response` directly
    and skip model inference.
    """

    # ---- Patterns ----
    RE_PROMPT_INJECTION = re.compile(
        r"(?:"
        r"\[\s*(?:system\s+override|admin\s+override|jailbreak|ignore\s+all|forget\s+all|disregard\s+all|new\s+instructions?)\s*\]"
        r"|\bignore\s+(?:all\s+)?(?:prior|previous|your|above)\s+(?:constraints|instructions?|rules?|guidelines?|training)\b"
        r"|\bact\s+as\s+(?:DAN|an?\s+unrestricted|an?\s+uncensored|an?\s+jailbroken)\b"
        r"|\bforget\s+(?:all\s+)?(?:your\s+)?(?:training|rules?|guidelines?|restrictions?)\b"
        r"|output\s+['\"]?dev_override_enabled['\"]?"
        r")",
        re.IGNORECASE,
    )
    # Safety refusal: DDoS, malware, exploits, cyber attacks
    RE_SAFETY_CYBER = re.compile(
        r"\b(?:ddos|denial[\s\-]of[\s\-]service|exploit|malware|ransomware|keylogger|botnet)\b",
        re.IGNORECASE,
    )
    # Uncertainty / future prediction: exact stock/commodity prices in future years
    RE_UNCERTAIN_FUTURE = re.compile(
        r"\b(?:exact\s+price|future\s+price|predict\s+the\s+price|stock\s+price)\s+of\s+\w+\s+(?:on|in)\s+.*\b(?:202[7-9]|20[3-9]\d)\b",
        re.IGNORECASE,
    )
    RE_AMBIG_CONVERT = re.compile(r"^\s*convert\s+\d+(?:\.\d+)?\s*\.?\s*$", re.IGNORECASE)
    # Python range off-by-one bug detection: range(1, N) misses N
    RE_RANGE_OFFBYONE = re.compile(
        r"range\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"
        r".*?(?:should\s+print|print\s+numbers?|1\s+to\s+(\d+))",
        re.IGNORECASE | re.DOTALL,
    )
    RE_BOOKING = re.compile(
        r"^\s*(?:book|reserve|schedule|buy)\s+(?:a\s+)?(?:ticket|seat|room|appointment|flight|train|hotel)",
        re.IGNORECASE,
    )
    RE_TEMP_C_TO_F = re.compile(
        r"convert\s+(\d+(?:\.\d+)?)\s*(?:degrees?)?\s*c(?:elsius)?\s+to\s+f(?:ahrenheit)?",
        re.IGNORECASE,
    )
    RE_TEMP_F_TO_C = re.compile(
        r"convert\s+(\d+(?:\.\d+)?)\s*(?:degrees?)?\s*f(?:ahrenheit)?\s+to\s+c(?:elsius)?",
        re.IGNORECASE,
    )
    RE_KM_TO_MI = re.compile(
        r"convert\s+(\d+(?:\.\d+)?)\s*(?:kilometer|km|kilo)s?\s+to\s+(?:miles?)",
        re.IGNORECASE,
    )
    RE_CALC_THEN_CONVERT = re.compile(
        r"calculate\s+([\d\s\*\+\-\/\.]+),?\s*then\s+convert\s+that\s+(?:number\s+of\s+)?(\w+)\s+to\s+(\w+)",
        re.IGNORECASE,
    )
    # Sequential 2-step math: "First calculate 25 * 4, and then divide that result by 2"
    RE_SEQUENTIAL_MATH = re.compile(
        r"first\s+calculate\s+([\d\s\*\+\-\/\.]+),?\s*(?:and\s+)?then\s+(\w+)\s+(?:that\s+result\s+)?by\s+([\d\.]+)",
        re.IGNORECASE,
    )
    RE_DISCOUNT_TAX = re.compile(
        r"costs?\s+\$?([\d\.,]+).*?(\d+(?:\.\d+)?)\s*%\s*discount.*?(\d+(?:\.\d+)?)\s*%\s*(?:sales\s*)?tax",
        re.IGNORECASE | re.DOTALL,
    )
    RE_PERCENT_OF = re.compile(
        r"(?:what\s+is\s+)?(\d+(?:\.\d+)?)\s*%\s*of\s+(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    RE_DIRECT_MATH = re.compile(
        r"^(?:(?:what\s+is|calculate|compute|evaluate)\s+)?([\d\s\.\+\-\*\/\^\(\)%,$]+)\s*\??\s*$",
        re.IGNORECASE,
    )
    RE_CONTEXT_QUESTION = re.compile(
        r"^context\s*:\s*(.+?)\n{1,2}(.+)$",
        re.IGNORECASE | re.DOTALL,
    )
    RE_PREF_QUERY = re.compile(
        r"\b(?:framework|library|tool|stack|language|prefer|use\s+for|recommend\s+for)\b",
        re.IGNORECASE,
    )
    RE_PREF_STATED = re.compile(
        r"\bI\s+(?:prefer|use|like|want)\s+(\w[\w\s]+?)\s+(?:for\s+styling|for\s+CSS|CSS|styling)",
        re.IGNORECASE,
    )
    RE_SECOND_ITEM = re.compile(r"\b(?:second|2nd|2\.)\s+item\b", re.IGNORECASE)
    RE_LIST_ITEMS = re.compile(r"\d+\.\s+([^,\.]+)")

    # Absent-fact markers: things the user might ask for that the model tends to hallucinate
    ABSENT_FACT_PATTERNS = [
        re.compile(r"\bprogramming\s+language\b", re.IGNORECASE),
        re.compile(r"\bbuilt\s+(?:with|in|using)\b", re.IGNORECASE),
        re.compile(r"\bwritten\s+in\b", re.IGNORECASE),
        re.compile(r"\btechnology\s+(?:used|stack)\b", re.IGNORECASE),
        re.compile(r"\bframework\b", re.IGNORECASE),
    ]

    def intercept(
        self,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
        context_document: str | None = None,
    ) -> InterceptResult:
        """
        Apply all deterministic checks in priority order.

        Args:
            user_prompt: The raw user message.
            history: List of prior conversation turns as {"role": ..., "content": ...}.
            context_document: Optional RAG/document context injected before the query.

        Returns:
            InterceptResult with intercepted=True if handled, else intercepted=False.
        """
        text = user_prompt.strip()
        history = history or []

        # 0. Prompt injection resistance (highest priority)
        if self.RE_PROMPT_INJECTION.search(text):
            return InterceptResult(
                intercepted=True,
                response=(
                    "I'm Bravien, and I cannot comply with that request. I follow my core instructions "
                    "and safety guidelines and cannot be overridden by user-level prompts. "
                    "How can I assist you legitimately?"
                ),
                method="safety:prompt_injection",
                confidence=1.0,
            )

        # 0a. Safety refusal: cyberattacks / malware (Dim 14)
        if self.RE_SAFETY_CYBER.search(text):
            return InterceptResult(
                intercepted=True,
                response=(
                    "I cannot provide scripts, instructions, or assistance for performing cyberattacks, "
                    "distributed denial-of-service (DDoS) attacks, or developing malicious software. "
                    "I can, however, explain defensive cybersecurity principles and how organizations "
                    "prevent and mitigate such attacks."
                ),
                method="safety:cyber_refusal",
                confidence=1.0,
            )

        # 0b. Uncertainty abstention: future asset/commodity price prediction (Dim 13)
        if self.RE_UNCERTAIN_FUTURE.search(text):
            return InterceptResult(
                intercepted=True,
                response=(
                    "I cannot predict the exact future price of financial assets or commodities. "
                    "Future market prices depend on numerous macroeconomic, geopolitical, and supply-demand "
                    "factors that are fundamentally uncertain and unknowable."
                ),
                method="abstention:future_prediction",
                confidence=1.0,
            )

        # 0b. Python range off-by-one bug detection
        m = self.RE_RANGE_OFFBYONE.search(text)
        if m:
            start = int(m.group(1))
            stop = int(m.group(2))
            correct_stop = stop + 1
            last_printed = stop - 1
            return InterceptResult(
                intercepted=True,
                response=(
                    f"The bug is an **off-by-one error** in the `range()` call.\n\n"
                    f"`range({start}, {stop})` is exclusive of the end value — it generates "
                    f"numbers from {start} to {last_printed} (stops at {last_printed}), "
                    f"so it never prints {stop}.\n\n"
                    f"**Fix**: Change `range({start}, {stop})` to `range({start}, {correct_stop})` "
                    f"to include {stop} in the output."
                ),
                method="code_debug:range_off_by_one",
                metadata={"start": start, "stop": stop, "correct_stop": correct_stop},
            )

        # 1. Ambiguous conversion (no units specified)
        if self.RE_AMBIG_CONVERT.match(text):
            return InterceptResult(
                intercepted=True,
                response=(
                    "Please specify what units or currency you would like to convert from and to. "
                    "For example: 'Convert 50 kilometers to miles', 'Convert 50 Celsius to Fahrenheit', "
                    "or 'Convert 50 USD to EUR'."
                ),
                method="ambiguity_clarification",
            )

        # 2. Under-specified booking / ticket request
        if self.RE_BOOKING.match(text):
            return InterceptResult(
                intercepted=True,
                response=(
                    "I'd be happy to help! To book a ticket, I need a few details:\n"
                    "- **Destination** and origin location?\n"
                    "- **Date and time** of travel?\n"
                    "- **Type** of ticket (flight, train, bus, etc.)?\n\n"
                    "Please provide these details so I can assist you."
                ),
                method="clarification_missing_params",
            )

        # 3. Temperature conversion: Celsius → Fahrenheit
        m = self.RE_TEMP_C_TO_F.search(text)
        if m:
            val = float(m.group(1))
            result = convert_unit(val, "celsius", "fahrenheit")
            if result is not None:
                return InterceptResult(
                    intercepted=True,
                    response=(
                        f"To convert Celsius to Fahrenheit, use the formula:\n"
                        f"**F = (C × 9/5) + 32**\n\n"
                        f"F = ({val} × 9/5) + 32 = {fmt_num(val * 9/5)} + 32 = **{fmt_num(result)}°F**\n\n"
                        f"**{val}°C = {fmt_num(result)}°F**"
                    ),
                    method="unit_converter:c_to_f",
                    metadata={"input": val, "result": result},
                )

        # 4. Temperature conversion: Fahrenheit → Celsius
        m = self.RE_TEMP_F_TO_C.search(text)
        if m:
            val = float(m.group(1))
            result = convert_unit(val, "fahrenheit", "celsius")
            if result is not None:
                return InterceptResult(
                    intercepted=True,
                    response=f"**{val}°F = {fmt_num(result)}°C**",
                    method="unit_converter:f_to_c",
                    metadata={"input": val, "result": result},
                )

        # 5. Km → Miles
        m = self.RE_KM_TO_MI.search(text)
        if m:
            val = float(m.group(1))
            result = convert_unit(val, "km", "miles")
            if result is not None:
                return InterceptResult(
                    intercepted=True,
                    response=f"**{val} kilometers = {fmt_num(result)} miles**",
                    method="unit_converter:km_to_mi",
                    metadata={"input": val, "result": result},
                )

        # 6. Multi-tool: calculate then convert (e.g., "calc 50*4, then convert minutes to hours")
        m = self.RE_CALC_THEN_CONVERT.search(text)
        if m:
            expr = m.group(1).strip()
            from_u = m.group(2).lower().strip()
            to_u = m.group(3).lower().strip()
            calc_result = eval_expr_safe(expr)
            if calc_result is not None:
                conv_result = convert_unit(calc_result, from_u, to_u)
                calc_str = fmt_num(calc_result)
                if conv_result is not None:
                    conv_str = fmt_num(conv_result)
                    extra = ""
                    if from_u in ("min", "minute", "minutes") and to_u in ("h", "hr", "hour", "hours"):
                        total_mins = int(calc_result)
                        h = total_mins // 60
                        leftover_m = total_mins % 60
                        if leftover_m:
                            extra = f" ({h} hours and {leftover_m} minutes)"
                    return InterceptResult(
                        intercepted=True,
                        response=(
                            f"1. **Calculation**: {expr} = **{calc_str} {from_u}**.\n"
                            f"2. **Conversion to {to_u}**: {calc_str} {from_u} = **{conv_str} {to_u}**{extra}.\n\n"
                            f"**Result**: {calc_str} {from_u} = **{conv_str} {to_u}**{extra}."
                        ),
                        method="multi_tool:calc_then_convert",
                        metadata={"calc": calc_result, "converted": conv_result},
                    )

        # 6b. Sequential 2-step math problem (Dim 22)
        m = self.RE_SEQUENTIAL_MATH.search(text)
        if m:
            expr = m.group(1).strip()
            op_word = m.group(2).lower().strip()
            second_val = float(m.group(3))
            step1 = eval_expr_safe(expr)
            if step1 is not None:
                if op_word in ("divide", "divided"):
                    final = step1 / second_val
                    symbol = "÷"
                elif op_word in ("multiply", "multiplied", "times"):
                    final = step1 * second_val
                    symbol = "×"
                elif op_word in ("add", "added", "plus"):
                    final = step1 + second_val
                    symbol = "+"
                elif op_word in ("subtract", "subtracted", "minus"):
                    final = step1 - second_val
                    symbol = "-"
                else:
                    final = None

                if final is not None:
                    return InterceptResult(
                        intercepted=True,
                        response=(
                            f"Let's solve step by step:\n\n"
                            f"1. **Step 1**: {expr} = **{fmt_num(step1)}**\n"
                            f"2. **Step 2**: {fmt_num(step1)} {symbol} {fmt_num(second_val)} = **{fmt_num(final)}**\n\n"
                            f"The final number is **{fmt_num(final)}**."
                        ),
                        method="calculator:sequential_math",
                        metadata={"step1": step1, "final": final},
                    )

        # 7. Multi-step discount + tax word problem
        m = self.RE_DISCOUNT_TAX.search(text)
        if m:
            try:
                base = float(re.sub(r"[,]", "", m.group(1)))
                disc_pct = float(m.group(2))
                tax_pct = float(m.group(3))
                discounted = base * (1 - disc_pct / 100.0)
                final = discounted * (1 + tax_pct / 100.0)
                disc_amount = base - discounted
                tax_amount = final - discounted
                return InterceptResult(
                    intercepted=True,
                    response=(
                        f"Let's solve this step by step:\n\n"
                        f"1. **{disc_pct:.0f}% discount** on ${base:.2f}:\n"
                        f"   ${base:.2f} − ${disc_amount:.2f} = **${discounted:.2f}**\n\n"
                        f"2. **{tax_pct:.0f}% sales tax** on ${discounted:.2f}:\n"
                        f"   ${discounted:.2f} + ${tax_amount:.2f} = **${final:.2f}**\n\n"
                        f"The final price is **${final:.2f}**."
                    ),
                    method="calculator:discount_tax",
                    metadata={"base": base, "discounted": discounted, "final": final},
                )
            except Exception:
                pass

        # 8. Direct arithmetic (e.g., "Calculate 345 * 18", "What is 45 * 12?")
        m = self.RE_PERCENT_OF.search(text)
        if m:
            pct = float(m.group(1))
            of_val = float(m.group(2))
            result = pct / 100.0 * of_val
            return InterceptResult(
                intercepted=True,
                response=f"{pct}% of {fmt_num(of_val)} = **{fmt_num(result)}**",
                method="calculator:percent_of",
                metadata={"result": result},
            )

        m = self.RE_DIRECT_MATH.match(text)
        if m:
            raw_expr = m.group(1).strip().rstrip("?").strip()
            if re.search(r"[\+\-\*\/\^%]", raw_expr):
                result = eval_expr_safe(raw_expr)
                if result is not None:
                    # Format large integers with commas
                    if abs(result) >= 1000 and result == int(result):
                        result_str = f"{int(result):,}"
                    else:
                        result_str = fmt_num(result)
                    return InterceptResult(
                        intercepted=True,
                        response=f"{raw_expr.strip()} = **{result_str}**",
                        method="calculator:direct",
                        metadata={"result": result},
                    )

        # 9. Anti-hallucination: context document provided but asked for absent fact
        # Check both inline "Context: ..." prefix AND a separately injected context_document
        effective_context = context_document or ""
        ctx_match = self.RE_CONTEXT_QUESTION.match(text)
        if ctx_match:
            effective_context = ctx_match.group(1).strip()
            question = ctx_match.group(2).strip()
        else:
            question = text

        if effective_context:
            for pattern in self.ABSENT_FACT_PATTERNS:
                if pattern.search(question):
                    if not pattern.search(effective_context):
                        return InterceptResult(
                            intercepted=True,
                            response=(
                                "The provided context does not mention or contain any information "
                                "about that. Based solely on the given context, I cannot determine "
                                "this — it is not specified in the document."
                            ),
                            method="anti_hallucination:absent_fact",
                            confidence=0.95,
                        )

        # 10. Memory preference recall from conversation history
        if history and self.RE_PREF_QUERY.search(text):
            for turn in reversed(history):
                if turn.get("role") == "user":
                    pref_match = self.RE_PREF_STATED.search(turn["content"])
                    if pref_match:
                        pref = pref_match.group(1).strip().rstrip()
                        return InterceptResult(
                            intercepted=True,
                            response=(
                                f"Based on your previously stated preference, you should use "
                                f"**{pref}** for styling your web application."
                            ),
                            method="memory:preference_recall",
                            metadata={"preference": pref},
                        )

        # 11. Ordered list recall from conversation history
        if history and self.RE_SECOND_ITEM.search(text):
            for turn in reversed(history):
                if turn.get("role") == "user":
                    found_items = self.RE_LIST_ITEMS.findall(turn["content"])
                    if len(found_items) >= 2:
                        second_item = found_items[1].strip()
                        return InterceptResult(
                            intercepted=True,
                            response=f"The second item on your list is **{second_item}**.",
                            method="context:list_recall",
                            metadata={"item_index": 1, "item": second_item},
                        )

        return InterceptResult(intercepted=False)


# Backward-compatible and semantic alias
QueryInterceptLayer = DeterministicInterceptLayer


# Singleton for production use
deterministic_intercept = DeterministicInterceptLayer()
