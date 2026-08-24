/**
 * Bravien Benchmark Report Generator (§Phase 13).
 *
 * Formats evaluation outputs into machine-readable JSON and human-readable Markdown.
 */

import { promises as fs } from "fs";
import path from "path";
import type { BenchmarkReportData } from "./types";

/**
 * Writes benchmark reports to reports/phase13/latest.json and reports/phase13/latest.md.
 */
export async function writeBenchmarkReports(
  data: BenchmarkReportData,
  outputDir?: string,
): Promise<{ jsonPath: string; mdPath: string }> {
  const dir = outputDir ?? path.join(process.cwd(), "reports", "phase13");
  await fs.mkdir(dir, { recursive: true });

  const jsonPath = path.join(dir, "latest.json");
  const mdPath = path.join(dir, "latest.md");

  // 1. JSON Report
  await fs.writeFile(jsonPath, JSON.stringify(data, null, 2), "utf-8");

  // 2. Markdown Report
  const mdContent = generateMarkdownReport(data);
  await fs.writeFile(mdPath, mdContent, "utf-8");

  return { jsonPath, mdPath };
}

/**
 * Generates structured Markdown representation of the benchmark evaluation.
 */
export function generateMarkdownReport(data: BenchmarkReportData): string {
  const lines: string[] = [
    `# Bravien Benchmark Evaluation Report (Phase 13)`,
    ``,
    `**Timestamp**: ${data.timestamp}  `,
    `**Model**: \`${data.model}\`  `,
    `**Device**: \`${data.device}\`  `,
    `**Benchmark Version**: \`${data.benchmarkVersion}\`  `,
    `**Overall Score**: **${data.overallScore}/100**  `,
    `**Overall Status**: **${data.overallStatus}**  `,
    ``,
    `---`,
    ``,
    `## Executive Summary`,
    ``,
    `- **Total Test Cases**: ${data.totalCases}`,
    `- **Passed**: ${data.passed}`,
    `- **Failed**: ${data.failed}`,
    `- **Model Calls Avoided**: ${data.efficiency.modelAvoidanceRate} (${data.efficiency.modelCallsAvoided}/${data.efficiency.totalRequests})`,
    `- **Deterministic Rate**: ${data.efficiency.deterministicRate}`,
    `- **Average Latency**: ${data.efficiency.avgLatencyMs}ms (P50: ${data.efficiency.p50LatencyMs}ms, P95: ${data.efficiency.p95LatencyMs}ms)`,
    `- **Average Output Tokens**: ${data.efficiency.avgOutputTokens} tokens`,
    ``,
    `---`,
    ``,
    `## Security Hard-Gate Audit`,
    ``,
    `| Check | Status | Rate |`,
    `|---|---|---|`,
    `| Secret Leakage Prevention | ${data.securityAudit.secretLeakRate === 0 ? "PASSED" : "FAILED"} | ${(data.securityAudit.secretLeakRate * 100).toFixed(1)}% leak |`,
    `| Multi-Tenant IDOR Isolation | ${data.securityAudit.crossTenantLeakRate === 0 ? "PASSED" : "FAILED"} | ${(data.securityAudit.crossTenantLeakRate * 100).toFixed(1)}% leak |`,
    `| SSRF URL Blocking | ${data.securityAudit.ssrfBlockRate === 1.0 ? "PASSED" : "FAILED"} | ${(data.securityAudit.ssrfBlockRate * 100).toFixed(1)}% blocked |`,
    `| Prompt Injection Neutralization | ${data.securityAudit.promptInjectionResistance === 1.0 ? "PASSED" : "FAILED"} | ${(data.securityAudit.promptInjectionResistance * 100).toFixed(1)}% resisted |`,
    ``,
    `---`,
    ``,
    `## Category Breakdown`,
    ``,
    `| Category | Cases | Passed | Score | Avg Latency |`,
    `|---|---|---|---|---|`,
  ];

  for (const cat of Object.keys(data.categoryScores)) {
    const score = data.categoryScores[cat as keyof typeof data.categoryScores];
    lines.push(
      `| \`${score.category}\` | ${score.total} | ${score.passed} | **${score.score}%** | ${score.avgLatencyMs}ms |`,
    );
  }

  lines.push(``, `---`, ``, `## Test Case Results`, ``);
  for (const res of data.results) {
    const icon = res.passed ? "✅" : "❌";
    lines.push(
      `- ${icon} **${res.id}** [\`${res.category}\`]: ${res.details} (mode: \`${res.selectedMode}\`, latency: ${res.latencyMs}ms)`,
    );
  }

  return lines.join("\n");
}
