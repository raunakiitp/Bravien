import { z } from "zod";
import type { ToolDefinition, ToolResult } from "./base";

const calculatorSchema = z.object({
  expression: z
    .string()
    .min(1, "Expression is required")
    .max(500, "Expression too long"),
});

/**
 * Evaluates basic arithmetic safely using recursive descent parsing.
 * Supports +, -, *, /, %, ^ (power), parentheses, and Math functions (sqrt, abs, sin, cos, tan, log, round, floor, ceil).
 */
function evaluateMath(expr: string): number {
  const sanitized = expr
    .replace(/×/g, "*")
    .replace(/÷/g, "/")
    .replace(/\s+/g, "");

  // Safe character check
  if (!/^[0-9+\-*/%^().a-zA-Z_]+$/.test(sanitized)) {
    throw new Error("Invalid characters in mathematical expression.");
  }

  let pos = 0;

  function peek(): string {
    return sanitized[pos] || "";
  }

  function get(): string {
    return sanitized[pos++] || "";
  }

  function parsePrimary(): number {
    const ch = peek();
    if (ch === "(") {
      get(); // consume '('
      const result = parseExpression();
      if (get() !== ")") throw new Error("Mismatched parentheses");
      return result;
    }

    if (ch === "-" || ch === "+") {
      const op = get();
      const val = parsePrimary();
      return op === "-" ? -val : val;
    }

    if (/[a-zA-Z_]/.test(ch)) {
      let name = "";
      while (/[a-zA-Z0-9_]/.test(peek())) {
        name += get();
      }
      if (peek() === "(") {
        get();
        const arg = parseExpression();
        if (get() !== ")") throw new Error(`Mismatched parentheses in ${name}`);
        const fn = name.toLowerCase();
        switch (fn) {
          case "sqrt":
            return Math.sqrt(arg);
          case "abs":
            return Math.abs(arg);
          case "sin":
            return Math.sin(arg);
          case "cos":
            return Math.cos(arg);
          case "tan":
            return Math.tan(arg);
          case "log":
            return Math.log10(arg);
          case "ln":
            return Math.log(arg);
          case "exp":
            return Math.exp(arg);
          case "round":
            return Math.round(arg);
          case "floor":
            return Math.floor(arg);
          case "ceil":
            return Math.ceil(arg);
          default:
            throw new Error(`Unsupported function: ${name}`);
        }
      }
      if (name.toLowerCase() === "pi") return Math.PI;
      if (name.toLowerCase() === "e") return Math.E;
      throw new Error(`Unknown identifier: ${name}`);
    }

    if (/[0-9.]/.test(ch)) {
      let numStr = "";
      while (/[0-9.]/.test(peek())) {
        numStr += get();
      }
      const num = Number(numStr);
      if (Number.isNaN(num)) throw new Error(`Invalid number: ${numStr}`);
      return num;
    }

    throw new Error(`Unexpected token '${ch}' at position ${pos}`);
  }

  function parsePower(): number {
    let left = parsePrimary();
    while (peek() === "^") {
      get();
      const right = parsePrimary();
      left = Math.pow(left, right);
    }
    return left;
  }

  function parseMultiplicative(): number {
    let left = parsePower();
    while (peek() === "*" || peek() === "/" || peek() === "%") {
      const op = get();
      const right = parsePower();
      if (op === "*") left *= right;
      else if (op === "/") {
        if (right === 0) throw new Error("Division by zero");
        left /= right;
      } else if (op === "%") {
        left %= right;
      }
    }
    return left;
  }

  function parseExpression(): number {
    let left = parseMultiplicative();
    while (peek() === "+" || peek() === "-") {
      const op = get();
      const right = parseMultiplicative();
      if (op === "+") left += right;
      else if (op === "-") left -= right;
    }
    return left;
  }

  const result = parseExpression();
  if (pos < sanitized.length) {
    throw new Error(`Unexpected extra characters: '${sanitized.slice(pos)}'`);
  }
  return result;
}

export const calculatorTool: ToolDefinition<typeof calculatorSchema> = {
  name: "calculator",
  description:
    "Evaluate standard arithmetic and mathematical expressions safely (supports +, -, *, /, %, ^, sqrt, abs, sin, cos, tan, log, ln, round, pi, e).",
  schema: calculatorSchema,
  execute: async ({ expression }) => {
    try {
      const result = evaluateMath(expression);
      return {
        success: true,
        data: { expression, result },
        formattedOutput: `Calculation Result: ${expression} = ${result}`,
      };
    } catch (err) {
      return {
        success: false,
        error: err instanceof Error ? err.message : String(err),
        formattedOutput: `Calculation Error: ${err instanceof Error ? err.message : "Failed to evaluate"}`,
      };
    }
  },
};
