/**
 * Centralized Intent Router for Bravien.
 *
 * Classifies user queries into semantic categories to determine:
 * 1. Primary intent
 * 2. Candidate tools
 * 3. Whether complex task planning is required
 * 4. Whether external web research is needed
 * 5. Detected entities / parameters
 */

export type UserIntent =
  | "GENERAL_CHAT"
  | "CALCULATION"
  | "TIME_DATE"
  | "MEMORY_QUERY"
  | "DOCUMENT_QUERY"
  | "PROJECT_QUERY"
  | "CONVERSATION_SEARCH"
  | "WEB_RESEARCH"
  | "CODING"
  | "DEBUGGING"
  | "SUMMARIZATION"
  | "COMPARISON"
  | "MULTI_STEP_TASK";

export interface IntentResult {
  intent: UserIntent;
  confidence: number;
  requiresPlanning: boolean;
  requiresWeb: boolean;
  candidateTools: string[];
  entities: Record<string, string>;
  reasoning: string;
}

export interface RouterContext {
  hasActiveProject?: boolean;
  hasWebAccess?: boolean;
}

const MATH_PATTERNS = [
  /^(?:calculate|compute|evaluate)\s+[0-9+\-*/%^().\s\w]+(?:=[?]?|[?]?)$/i,
  /^what is\s+(?=.*[0-9])[0-9+\-*/%^().\s\w]+(?:=[?]?|[?]?)$/i,
  /^([0-9+\-*/%^().\s]+[+\-*/%^][0-9+\-*/%^().\s]+)$/,
  /\b(?:sqrt|sin|cos|tan|log|factorial)\s*\([0-9.]+\)/i,
];

const TIME_PATTERNS = [
  /\b(?:what time is it|current time|today's date|what is today's date|what is the date|what day is it|current date and time|what is the current date and time|current date|what's the time|what's today's date)\b/i,
  /\b(?:date and time|time and date)\b/i,
];

const CODING_PATTERNS = [
  /\b(?:write|create|implement|generate|refactor|optimize)\s+(?:a\s+)?(?:python|typescript|javascript|html|css|sql|rust|go|c\+\+|bash|react|next\.?js)?\s*(?:script|function|class|method|component|api|query|regex|algorithm)\b/i,
  /\b(?:code snippet|fenced code|function to|class to|syntax for)\b/i,
  /```[\w]*\n[\s\S]*?\n```/,
];

const DEBUG_PATTERNS = [
  /\b(?:fix this error|why is this failing|traceback|syntaxerror|typeerror|referenceerror|nullpointer|bug in|debug this)\b/i,
  /\b(?:error:\s|exception:\s|failed with exit code)\b/i,
];

const WEB_PATTERNS = [
  /\b(?:search the web for|look up online|latest news on|current weather|recent updates on|what happened today with|who won the|release notes for|what changed in)\b/i,
  /\b(?:search online|find on the web|browse the web for|google for)\b/i,
  /\b(?:today's news|latest release of|current version of|what are the latest|latest features of|latest\s+\w+\s+features)\b/i,
];

const COMPARISON_PATTERNS = [
  /\b(?:compare\s+.+\s+with\s+|difference between\s+.+\s+and\s+|\bvs\b|\bversus\b|pros and cons of\s+.+\s+and\s+|compare\s+.+\s+and\s+)/i,
];

const MULTI_STEP_PATTERNS = [
  /\b(?:first.+then.+finally|step by step plan|research\s+.+(?:compare|pros and cons|analyze)|plan and implement|analyze and generate a report)\b/i,
  /\b(?:research\s+.+\s*,?\s*compare\s+)/i,
];

export function routeIntent(
  query: string,
  context: RouterContext = {},
): IntentResult {
  const q = query.trim();
  const lower = q.toLowerCase();

  // 1. Date & Time
  for (const pattern of TIME_PATTERNS) {
    if (pattern.test(lower)) {
      return {
        intent: "TIME_DATE",
        confidence: 0.95,
        requiresPlanning: false,
        requiresWeb: false,
        candidateTools: ["get_current_time"],
        entities: {},
        reasoning: "Matched date/time query pattern.",
      };
    }
  }

  // 2. Math / Calculation
  for (const pattern of MATH_PATTERNS) {
    if (pattern.test(q)) {
      return {
        intent: "CALCULATION",
        confidence: 0.95,
        requiresPlanning: false,
        requiresWeb: false,
        candidateTools: ["calculator"],
        entities: { rawQuery: q },
        reasoning: "Matched mathematical calculation pattern.",
      };
    }
  }

  // 3. Multi-Step Task / Research & Compare
  for (const pattern of MULTI_STEP_PATTERNS) {
    if (pattern.test(lower)) {
      return {
        intent: "MULTI_STEP_TASK",
        confidence: 0.9,
        requiresPlanning: true,
        requiresWeb: context.hasWebAccess ?? true,
        candidateTools: ["search_web", "search_documents", "calculator"],
        entities: { taskGoal: q },
        reasoning: "Identified multi-phase goal requiring structured execution.",
      };
    }
  }

  // 4. Comparison
  for (const pattern of COMPARISON_PATTERNS) {
    if (pattern.test(lower)) {
      const isMultiStep = lower.length > 50 || lower.includes("research") || lower.includes("recommend");
      return {
        intent: "COMPARISON",
        confidence: 0.85,
        requiresPlanning: isMultiStep,
        requiresWeb: lower.includes("latest") || lower.includes("recent"),
        candidateTools: isMultiStep ? ["search_web", "search_documents"] : [],
        entities: { comparisonQuery: q },
        reasoning: "Comparative analysis requested.",
      };
    }
  }

  // 5. Explicit Web Research
  for (const pattern of WEB_PATTERNS) {
    if (pattern.test(lower)) {
      return {
        intent: "WEB_RESEARCH",
        confidence: 0.9,
        requiresPlanning: lower.length > 80 || lower.includes("and summarize") || lower.includes("and compare"),
        requiresWeb: true,
        candidateTools: ["search_web"],
        entities: { searchQuery: q },
        reasoning: "Query seeks live/recent online information.",
      };
    }
  }

  // 6. Memory Query
  if (
    lower.startsWith("search memories for") ||
    lower.startsWith("do i remember") ||
    lower.startsWith("what are my preferences for") ||
    lower.startsWith("check my memories for") ||
    lower.includes("what did i tell you about")
  ) {
    return {
      intent: "MEMORY_QUERY",
      confidence: 0.9,
      requiresPlanning: false,
      requiresWeb: false,
      candidateTools: ["search_memories"],
      entities: {
        memoryQuery: q.replace(/^(?:search memories for|do i remember|what are my preferences for|check my memories for)\s*/i, "").trim(),
      },
      reasoning: "Direct lookup in personal memory store.",
    };
  }

  // 7. Project & Document Query
  if (
    context.hasActiveProject &&
    (lower.startsWith("search documents for") ||
      lower.startsWith("search files for") ||
      lower.includes("in the project documents") ||
      lower.includes("according to the uploaded") ||
      lower.includes("uploaded document") ||
      lower.includes("uploaded system document") ||
      lower.includes("what does the uploaded") ||
      lower.includes("what does the document say about"))
  ) {
    return {
      intent: "DOCUMENT_QUERY",
      confidence: 0.9,
      requiresPlanning: lower.includes("summarize all") || lower.includes("compare all"),
      requiresWeb: false,
      candidateTools: ["search_documents", "get_document_content"],
      entities: {
        docQuery: q.replace(/^(?:search documents for|search files for)\s*/i, "").trim(),
      },
      reasoning: "Targeted document search within active workspace project.",
    };
  }

  // 8. Conversation History Search
  if (
    lower.startsWith("search conversations for") ||
    lower.startsWith("find in past chats") ||
    lower.startsWith("find earlier discussion on")
  ) {
    return {
      intent: "CONVERSATION_SEARCH",
      confidence: 0.9,
      requiresPlanning: false,
      requiresWeb: false,
      candidateTools: ["search_conversations"],
      entities: {
        chatQuery: q.replace(/^(?:search conversations for|find in past chats|find earlier discussion on)\s*/i, "").trim(),
      },
      reasoning: "Searching previous chat messages.",
    };
  }

  // 9. Debugging / Troubleshooting
  for (const pattern of DEBUG_PATTERNS) {
    if (pattern.test(q)) {
      return {
        intent: "DEBUGGING",
        confidence: 0.85,
        requiresPlanning: false,
        requiresWeb: false,
        candidateTools: [],
        entities: {},
        reasoning: "Error traceback or debugging investigation.",
      };
    }
  }

  // 10. Coding / Implementation
  for (const pattern of CODING_PATTERNS) {
    if (pattern.test(q)) {
      return {
        intent: "CODING",
        confidence: 0.85,
        requiresPlanning: false,
        requiresWeb: false,
        candidateTools: [],
        entities: {},
        reasoning: "Software implementation or code generation inquiry.",
      };
    }
  }

  // 11. Summarization
  if (
    lower.startsWith("summarize the following") ||
    lower.startsWith("summarize this") ||
    lower.startsWith("tl;dr") ||
    lower.includes("give me a summary of")
  ) {
    return {
      intent: "SUMMARIZATION",
      confidence: 0.85,
      requiresPlanning: false,
      requiresWeb: false,
      candidateTools: [],
      entities: {},
      reasoning: "Text summarization requested.",
    };
  }

  // Default: General Chat
  return {
    intent: "GENERAL_CHAT",
    confidence: 0.7,
    requiresPlanning: false,
    requiresWeb: false,
    candidateTools: [],
    entities: {},
    reasoning: "Standard conversational inquiry.",
  };
}
