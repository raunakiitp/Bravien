/**
 * Bravien Evaluation Feedback Loop (§Phase 14).
 *
 * Reads Phase 13 benchmark evaluation reports and derives advisory improvement
 * signals to highlight category gaps, routing discrepancies, and model avoidance opportunities.
 */

import { promises as fs } from "fs";
import path from "path";
import type { AdvisoryImprovementSignal } from "./types";
import type { BenchmarkReportData } from "../evaluation/types";

/**
 * Extracts advisory improvement signals from a benchmark report.
 */
export async function deriveImprovementSignals(
  reportPathOrData?: string | BenchmarkReportData,
): Promise<AdvisoryImprovementSignal[]> {
  let data: BenchmarkReportData | null = null;

  if (typeof reportPathOrData === "object" && reportPathOrData !== null) {
    data = reportPathOrData;
  } else {
    const filePath =
      reportPathOrData ?? path.join(process.cwd(), "reports", "phase13", "latest.json");
    try {
      const content = await fs.readFile(filePath, "utf-8");
      data = JSON.parse(content) as BenchmarkReportData;
    } catch {
      return [];
    }
  }

  if (!data || !data.categoryScores) {
    return [];
  }

  const signals: AdvisoryImprovementSignal[] = [];

  for (const cat of Object.keys(data.categoryScores)) {
    const score = data.categoryScores[cat as keyof typeof data.categoryScores];
    if (score && score.score < 100) {
      const failureCount = score.total - score.passed;
      let suggestedAction = `Review ${cat} routing and response synthesis.`;

      if (cat === "TOOL_SELECTION") {
        suggestedAction = "Enhance deterministic tool intent keywords and calculator regex patterns.";
      } else if (cat === "DOCUMENT_RAG") {
        suggestedAction = "Calibrate RAG relevance threshold and citation coordinate formatting.";
      } else if (cat === "CODING") {
        suggestedAction = "Allocate CODE profile with larger maxTokens budget (1536 tokens).";
      } else if (cat === "MULTI_STEP_REASONING") {
        suggestedAction = "Refine planner step decomposition bounds and dependency execution order.";
      }

      signals.push({
        category: cat,
        failureCount,
        suggestedAction,
        confidence: Math.round(((100 - score.score) / 100) * 100) / 100,
      });
    }
  }

  return signals;
}
