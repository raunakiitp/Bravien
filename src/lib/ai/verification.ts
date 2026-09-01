/**
 * Lightweight Risk-Based Self-Verification & Anti-Hallucination Layer for Bravien.
 *
 * Evaluates high-risk generated outputs (e.g. arithmetic, code structure, document citations,
 * uncertainty claims) without incurring token overhead on trivial conversational turns.
 */

export interface VerificationResult {
  verified: boolean;
  score: number; // 0.0 - 1.0
  reason?: string;
  adjustedText?: string;
}

/**
 * Validates whether a generated response for an arithmetic expression matches deterministic calculation.
 */
export function verifyArithmeticResponse(prompt: string, response: string): VerificationResult {
  const mathMatch = prompt.match(/([0-9]+(?:\.[0-9]+)?)\s*([+\-*/])\s*([0-9]+(?:\.[0-9]+)?)/);
  if (!mathMatch) {
    return { verified: true, score: 1.0 };
  }

  const num1 = parseFloat(mathMatch[1]);
  const op = mathMatch[2];
  const num2 = parseFloat(mathMatch[3]);

  let expected: number;
  switch (op) {
    case "+":
      expected = num1 + num2;
      break;
    case "-":
      expected = num1 - num2;
      break;
    case "*":
      expected = num1 * num2;
      break;
    case "/":
      expected = num2 !== 0 ? num1 / num2 : NaN;
      break;
    default:
      return { verified: true, score: 1.0 };
  }

  if (isNaN(expected)) return { verified: true, score: 1.0 };

  const expectedStr = Number.isInteger(expected) ? expected.toString() : expected.toFixed(2);
  const containsExpected = response.includes(expectedStr);

  if (containsExpected) {
    return { verified: true, score: 1.0 };
  }

  return {
    verified: false,
    score: 0.2,
    reason: `Model calculation deviation. Expected ${expectedStr} for ${prompt}.`,
    adjustedText: `The result of ${num1} ${op} ${num2} is **${expectedStr}**.`,
  };
}

/**
 * Validates document grounding: ensures claims reflect provided document chunks or abstain honestly.
 */
export function verifyDocumentGrounding(
  documentContext: string,
  response: string,
): VerificationResult {
  if (!documentContext || !documentContext.trim()) {
    return { verified: true, score: 1.0 };
  }

  const lowerDoc = documentContext.toLowerCase();
  const lowerRes = response.toLowerCase();

  // Honest abstention is valid
  const isAbstention =
    lowerRes.includes("not contain sufficient information") ||
    lowerRes.includes("does not contain") ||
    lowerRes.includes("not found in the document") ||
    lowerRes.includes("no information provided");

  if (isAbstention) {
    return { verified: true, score: 1.0, reason: "Honest abstention acknowledged." };
  }

  // Extract key capitalized terms or numerical facts from response
  const responseFacts = response.match(/[A-Z][a-z]+|[0-9]{4}|[0-9]+(?:\.[0-9]+)?/g) ?? [];
  let matchingFacts = 0;

  for (const fact of responseFacts) {
    if (lowerDoc.includes(fact.toLowerCase())) {
      matchingFacts++;
    }
  }

  const groundingRatio = responseFacts.length > 0 ? matchingFacts / responseFacts.length : 1.0;

  return {
    verified: groundingRatio >= 0.3 || isAbstention,
    score: groundingRatio,
    reason: `Document grounding ratio: ${(groundingRatio * 100).toFixed(1)}%`,
  };
}

/**
 * Performs lightweight structural & syntax validation for generated code.
 */
export function verifyCodeSyntax(code: string, language = "python"): VerificationResult {
  if (!code || !code.trim()) {
    return { verified: false, score: 0.0, reason: "Empty code generated." };
  }

  const lines = code.split("\n");
  let openBrackets = 0;
  let openBraces = 0;
  let openParens = 0;

  for (const char of code) {
    if (char === "[") openBrackets++;
    if (char === "]") openBrackets--;
    if (char === "{") openBraces++;
    if (char === "}") openBraces--;
    if (char === "(") openParens++;
    if (char === ")") openParens--;
  }

  const balanced = openBrackets === 0 && openBraces === 0 && openParens === 0;

  if (!balanced) {
    return {
      verified: false,
      score: 0.5,
      reason: `Unbalanced syntax delimiter in generated ${language} code.`,
    };
  }

  if (language === "python") {
    // Check indentation consistency
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (line.endsWith(":") && i + 1 < lines.length) {
        const nextLine = lines[i + 1];
        if (nextLine.trim() && !nextLine.startsWith(" ") && !nextLine.startsWith("\t")) {
          return {
            verified: false,
            score: 0.6,
            reason: `Missing indentation after colon on line ${i + 1}.`,
          };
        }
      }
    }
  }

  return { verified: true, score: 1.0, reason: "Code structural syntax valid." };
}
