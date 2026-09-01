/**
 * Bravien Self-Improvement & Failure Recovery Types (§Phase 14).
 *
 * Defines failure categories, recovery strategies, attempt telemetry,
 * strategy memory structures, and advisory improvement signals.
 */

export type FailureCategory =
  | "ROUTING_FAILURE"
  | "TOOL_FAILURE"
  | "RETRIEVAL_FAILURE"
  | "MODEL_FAILURE"
  | "CONTEXT_FAILURE"
  | "PLANNING_FAILURE"
  | "AUTHORIZATION_FAILURE"
  | "VALIDATION_FAILURE"
  | "SECURITY_FAILURE"
  | "TIMEOUT"
  | "UNKNOWN";

export interface FailureAnalysisInput {
  executionMode: string;
  errorCode?: string;
  errorMessage: string;
  failedTool?: string;
  failedStep?: number;
  requestCategory?: string;
  modelCalled?: boolean;
  webResearchAttempted?: boolean;
  ragAttempted?: boolean;
}

export interface FailureAnalysisResult {
  category: FailureCategory;
  retryable: boolean;
  alternativeMode?: string;
  reason: string;
}

export interface RecoveryStrategy {
  id: string;
  description: string;
  targetMode: string;
  fallbackProfile?: "FAST" | "BALANCED" | "CREATIVE" | "CODE";
  maxTokens: number;
  useDeterministicShortcut?: boolean;
}

export interface RecoveryAttempt {
  originalMode: string;
  attemptedMode: string;
  failureCategory: FailureCategory;
  reason: string;
  startedAt: string;
  completedAt: string;
  success: boolean;
  tokensUsed?: number;
  latencyMs?: number;
}

export interface StrategyRecord {
  queryFingerprint: string;
  category: string;
  successfulMode: string;
  toolUsed?: string;
  retrievalSource?: "RAG" | "WEB" | "NONE";
  timestamp: number;
  userId?: string;
  projectId?: string;
}

export interface AdvisoryImprovementSignal {
  category: string;
  failureCount: number;
  suggestedAction: string;
  confidence: number;
}
