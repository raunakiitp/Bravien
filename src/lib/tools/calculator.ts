import { z } from "zod";

const calculatorParams = z.object({
  expression: z
    .string()
    .describe(
      "A basic arithmetic expression using numbers and + - * / % ( ) only",
    ),
});

/** Whitelist: digits, whitespace, decimal points, and + - * / % ( ) */
const SAFE_EXPR = /^[\d\s+\-*/().%]+$/;

/**
 * Evaluate basic arithmetic safely without mathjs.
 * Rejects identifiers, assignments, and any non-whitelisted characters.
 */
export function evaluateArithmetic(expression: string): number {
  const cleaned = expression.replace(/\s+/g, "");
  if (!cleaned) {
    throw new Error("Empty expression");
  }
  if (!SAFE_EXPR.test(cleaned)) {
    throw new Error(
      "Expression contains unsupported characters. Only numbers and + - * / % ( ) are allowed.",
    );
  }
  if (/[a-zA-Z_$]/.test(cleaned)) {
    throw new Error("Identifiers are not allowed");
  }

  // Intentional: the expression was already reduced to a digit/operator
  // whitelist above, and identifiers were rejected, so nothing callable remains.
  const fn = new Function(`"use strict"; return (${cleaned});`);
  const result = fn();
  if (typeof result !== "number" || !Number.isFinite(result)) {
    throw new Error("Expression did not evaluate to a finite number");
  }
  return result;
}

export const calculatorTool = {
  name: "calculator" as const,
  description:
    "Evaluate a basic arithmetic expression (+ - * / % and parentheses). Use for precise math.",
  parameters: calculatorParams,
  async execute(args: unknown) {
    const parsed = calculatorParams.parse(args);
    try {
      const value = evaluateArithmetic(parsed.expression);
      return { expression: parsed.expression, result: value };
    } catch (err) {
      return {
        expression: parsed.expression,
        error: err instanceof Error ? err.message : "Evaluation failed",
      };
    }
  },
};
