/**
 * Bravien Evaluation Framework Types (§Phase 13).
 *
 * Defines test case schemas, category taxonomies, execution modes,
 * scoring models, telemetry records, and benchmark report structures.
 */

export type EvalCategory =
  | "SIMPLE_FACTUAL"
  | "CALCULATION"
  | "TIME_DATE"
  | "MEMORY"
  | "DOCUMENT_RAG"
  | "WEB_RESEARCH"
  | "MULTI_STEP_REASONING"
  | "TASK_EXECUTION"
  | "AGENT_STATE"
  | "CODING"
  | "SECURITY"
  | "CONVERSATION_CONTEXT"
  | "TOOL_SELECTION"
  | "GENERAL_EXPLANATION";

export type EvalDifficulty = "EASY" | "MEDIUM" | "HARD";

export interface TestCase {
  id: string;
  category: EvalCategory;
  prompt: string;
  expectedMode: "DIRECT" | "TOOL" | "RESEARCH" | "PLANNED" | "TASK" | "WAITING_CONFIRMATION";
  expectedTools: string[];
  expectedBehavior: string;
  difficulty: EvalDifficulty;
  requiresModel: boolean;
  deterministic: boolean;
  shouldAbstain?: boolean;
  securityAttackType?: "PROMPT_INJECTION" | "SSRF" | "PATH_TRAVERSAL" | "IDOR" | "SECRET_LEAK";
  validationKeywords?: string[];
  forbiddenKeywords?: string[];
}

export interface TestCaseResult {
  id: string;
  category: EvalCategory;
  passed: boolean;
  selectedMode: string;
  selectedTools: string[];
  modelAvoided: boolean;
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
  error?: string;
  securityViolation?: boolean;
  hallucinationDetected?: boolean;
  details: string;
}

export interface CategoryScore {
  category: EvalCategory;
  total: number;
  passed: number;
  score: number; // 0 - 100
  avgLatencyMs: number;
}

export interface SecurityAuditResult {
  passed: boolean;
  secretLeakRate: number;
  crossTenantLeakRate: number;
  promptInjectionResistance: number;
  ssrfBlockRate: number;
  status: "PASSED" | "CRITICAL_FAILURE";
}

export interface BenchmarkReportData {
  timestamp: string;
  benchmarkVersion: string;
  model: string;
  device: string;
  totalCases: number;
  passed: number;
  failed: number;
  overallScore: number;
  overallStatus: "PASSED" | "CRITICAL_FAILURE";
  categoryScores: Record<EvalCategory, CategoryScore>;
  securityAudit: SecurityAuditResult;
  efficiency: {
    totalRequests: number;
    modelCalls: number;
    modelCallsAvoided: number;
    modelAvoidanceRate: string;
    deterministicRate: string;
    cacheHitRate: string;
    avgInputTokens: number;
    avgOutputTokens: number;
    avgLatencyMs: number;
    p50LatencyMs: number;
    p95LatencyMs: number;
  };
  results: TestCaseResult[];
}
