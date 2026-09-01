/**
 * Bravien Stage 8: Unified Intent Classification Layer.
 *
 * Provides deterministic and heuristic classification for incoming user turns.
 */

export type IntentCategory =
  | "conversation"
  | "factual_question"
  | "reasoning"
  | "mathematics"
  | "coding"
  | "document_question"
  | "memory"
  | "planning"
  | "tool_required"
  | "multi_step_task"
  | "clarification_required"
  | "unsafe_request"
  | "unknown";

export interface IntentResult {
  intent: IntentCategory;
  confidence: number;
  requiresTool: boolean;
  requiresMemory: boolean;
  requiresRag: boolean;
  requiresReasoning: boolean;
  requiresConfirmation: boolean;
  safetyLevel: "SAFE" | "CAUTION" | "UNSAFE";
  targetTool?: string;
  extractedEntities?: Record<string, unknown>;
  reasoning: string;
}

export interface TaskSpec {
  intent: IntentCategory;
  complexity: "LOW" | "MEDIUM" | "HIGH";
  requiresModel: boolean;
  requiresTool: boolean;
  requiresMemory: boolean;
  requiresRag: boolean;
  requiresVerification: boolean;
  riskLevel: "LOW" | "MEDIUM" | "HIGH";
  expectedOutput: string;
  maxSteps: number;
  timeoutSeconds: number;
  confirmationRequired: boolean;
}

const UNSAFE_PATTERNS = [
  /\b(ddos|dos attack|syn flood|botnet|exploit payload)\b/i,
  /\b(phishing email|keylogger|malware script|credential harvest|ransomware)\b/i,
  /\b(rm\s+-rf\s+[/~]|drop\s+database|format\s+c:)\b/i,
];

const MATH_PATTERNS = [
  /^\s*(\d+(\.\d+)?\s*[\+\-\*\/\^%]\s*)+\d+(\.\d+)?\s*\??\s*$/,
  /^\s*what\s+is\s+([0-9\.\s\+\-\*\/\^\(\)%]+)\??\s*$/i,
  /^\s*calculate\s+([0-9\.\s\+\-\*\/\^\(\)%]+)\??\s*$/i,
  /^\s*(\d+)%\s+of\s+(\d+)\??\s*$/i,
];

const MEMORY_WRITE_PATTERNS = [
  /\b(remember\s+(that)?|my\s+(favorite|preferred|name|email|budget|setting)\s+is)\b/i,
  /\b(save\s+(this\s+to\s+)?memory|store\s+(this\s+preference)?)\b/i,
  /\b(i\s+prefer\s+[\w\s]+|keep\s+in\s+mind\s+that)\b/i,
];

const MEMORY_READ_PATTERNS = [
  /\b(what\s+is\s+my\s+(favorite|preferred|budget|name))\b/i,
  /\b(what\s+did\s+i\s+(say|tell\s+you)|do\s+you\s+remember\s+my)\b/i,
];

const DOCUMENT_PATTERNS = [
  /\b(in\s+this\s+(document|pdf|file|context|article))\b/i,
  /\b(according\s+to\s+the\s+(document|pdf|file|context|text))\b/i,
  /\b(summarize\s+(this|the)\s+(document|pdf|file))\b/i,
];

const PLANNING_PATTERNS = [
  /\b(plan\s+(my|a)|step-by-step\s+plan|break\s+down\s+the\s+steps|roadmap)\b/i,
  /\b(create\s+a\s+migration\s+plan|study\s+schedule)\b/i,
];

const CODING_PATTERNS = [
  /\b(write|create|implement|debug|refactor)\s+(a\s+)?(python|javascript|typescript|rust|c\+\+|sql|html|css|function|class|algorithm|script)\b/i,
  /\b(time\s+complexity|space\s+complexity|big\s*-?\s*o|palindrome\s+function)\b/i,
];

const GREETINGS = [
  /^(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|sup)\b/i,
  /^(how\s+are\s+you|how's\s+it\s+going|what's\s+up|thank\s+you|thanks)\b/i,
  /^(who\s+are\s+you|what\s+is\s+your\s+name)\??$/i,
];

export function classifyIntent(query: string): IntentResult {
  const text = (query || "").trim();
  if (!text) {
    return {
      intent: "unknown",
      confidence: 1.0,
      requiresTool: false,
      requiresMemory: false,
      requiresRag: false,
      requiresReasoning: false,
      requiresConfirmation: false,
      safetyLevel: "SAFE",
      reasoning: "Empty input",
    };
  }

  // 1. Safety
  for (const pat of UNSAFE_PATTERNS) {
    if (pat.test(text)) {
      return {
        intent: "unsafe_request",
        confidence: 1.0,
        requiresTool: false,
        requiresMemory: false,
        requiresRag: false,
        requiresReasoning: false,
        requiresConfirmation: false,
        safetyLevel: "UNSAFE",
        reasoning: "Detected unsafe cyber-attack or system command pattern",
      };
    }
  }

  // 2. Math
  for (const pat of MATH_PATTERNS) {
    if (pat.test(text)) {
      return {
        intent: "mathematics",
        confidence: 0.98,
        requiresTool: true,
        requiresMemory: false,
        requiresRag: false,
        requiresReasoning: true,
        requiresConfirmation: false,
        safetyLevel: "SAFE",
        targetTool: "calculator",
        reasoning: "Deterministic arithmetic expression",
      };
    }
  }

  // 3. Memory
  for (const pat of MEMORY_WRITE_PATTERNS) {
    if (pat.test(text)) {
      return {
        intent: "memory",
        confidence: 0.95,
        requiresTool: true,
        requiresMemory: true,
        requiresRag: false,
        requiresReasoning: false,
        requiresConfirmation: false,
        safetyLevel: "SAFE",
        targetTool: "memory_write",
        reasoning: "Explicit user preference or memory store directive",
      };
    }
  }

  for (const pat of MEMORY_READ_PATTERNS) {
    if (pat.test(text)) {
      return {
        intent: "memory",
        confidence: 0.95,
        requiresTool: true,
        requiresMemory: true,
        requiresRag: false,
        requiresReasoning: false,
        requiresConfirmation: false,
        safetyLevel: "SAFE",
        targetTool: "memory_read",
        reasoning: "User preference or historical fact recall request",
      };
    }
  }

  // 4. Document / RAG
  for (const pat of DOCUMENT_PATTERNS) {
    if (pat.test(text)) {
      return {
        intent: "document_question",
        confidence: 0.92,
        requiresTool: true,
        requiresMemory: false,
        requiresRag: true,
        requiresReasoning: true,
        requiresConfirmation: false,
        safetyLevel: "SAFE",
        targetTool: "document_retrieval",
        reasoning: "Document or PDF context inquiry",
      };
    }
  }

  // 5. Planning
  for (const pat of PLANNING_PATTERNS) {
    if (pat.test(text)) {
      return {
        intent: "planning",
        confidence: 0.90,
        requiresTool: true,
        requiresMemory: false,
        requiresRag: false,
        requiresReasoning: true,
        requiresConfirmation: false,
        safetyLevel: "SAFE",
        targetTool: "structured_planning",
        reasoning: "Planning and task decomposition request",
      };
    }
  }

  // 6. Coding
  for (const pat of CODING_PATTERNS) {
    if (pat.test(text)) {
      return {
        intent: "coding",
        confidence: 0.88,
        requiresTool: false,
        requiresMemory: false,
        requiresRag: false,
        requiresReasoning: true,
        requiresConfirmation: false,
        safetyLevel: "SAFE",
        reasoning: "Programming or algorithm query",
      };
    }
  }

  // 7. Conversation
  for (const pat of GREETINGS) {
    if (pat.test(text)) {
      return {
        intent: "conversation",
        confidence: 0.95,
        requiresTool: false,
        requiresMemory: false,
        requiresRag: false,
        requiresReasoning: false,
        requiresConfirmation: false,
        safetyLevel: "SAFE",
        reasoning: "Social greeting or persona inquiry",
      };
    }
  }

  // 8. Ambiguous conversions
  if (/^\s*convert\s+\d+(\.\d+)?\s*\.?\s*$/i.test(text)) {
    return {
      intent: "clarification_required",
      confidence: 0.98,
      requiresTool: false,
      requiresMemory: false,
      requiresRag: false,
      requiresReasoning: false,
      requiresConfirmation: false,
      safetyLevel: "SAFE",
      reasoning: "Ambiguous conversion request lacking source and target units",
    };
  }

  // 9. Factual
  if (/^(what|when|where|who|which|why|how)\b/i.test(text)) {
    return {
      intent: "factual_question",
      confidence: 0.80,
      requiresTool: false,
      requiresMemory: false,
      requiresRag: false,
      requiresReasoning: false,
      requiresConfirmation: false,
      safetyLevel: "SAFE",
      reasoning: "Informational factual query",
    };
  }

  return {
    intent: "conversation",
    confidence: 0.70,
    requiresTool: false,
    requiresMemory: false,
    requiresRag: false,
    requiresReasoning: false,
    requiresConfirmation: false,
    safetyLevel: "SAFE",
    reasoning: "General conversational discourse",
  };
}

export function buildTaskSpec(userMessage: string, intentResult?: IntentResult): TaskSpec {
  const res = intentResult || classifyIntent(userMessage);

  let complexity: "LOW" | "MEDIUM" | "HIGH" = "LOW";
  if (res.intent === "multi_step_task" || res.intent === "planning") {
    complexity = "HIGH";
  } else if (res.intent === "coding" || res.intent === "reasoning" || res.intent === "mathematics") {
    complexity = "MEDIUM";
  }

  let riskLevel: "LOW" | "MEDIUM" | "HIGH" = "LOW";
  if (res.safetyLevel === "UNSAFE") {
    riskLevel = "HIGH";
  } else if (res.requiresConfirmation) {
    riskLevel = "MEDIUM";
  }

  return {
    intent: res.intent,
    complexity,
    requiresModel: res.intent !== "mathematics" && res.intent !== "clarification_required",
    requiresTool: res.requiresTool,
    requiresMemory: res.requiresMemory,
    requiresRag: res.requiresRag,
    requiresVerification: res.intent === "mathematics" || res.intent === "coding" || res.intent === "document_question",
    riskLevel,
    expectedOutput: complexity === "LOW" ? "Direct answer" : "Structured response",
    maxSteps: complexity === "HIGH" ? 8 : 4,
    timeoutSeconds: 30.0,
    confirmationRequired: res.requiresConfirmation,
  };
}
