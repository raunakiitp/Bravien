import { runBenchmark } from "../src/lib/ai/evaluation/runner";
import { writeBenchmarkReports } from "../src/lib/ai/evaluation/report";

async function main() {
  console.log("==================================================");
  console.log("BRAVIEN BENCHMARK RUNNER (PHASE 13)");
  console.log("==================================================\n");

  console.log("Running 55 evaluation test cases across 14 categories...");
  const report = await runBenchmark();
  const { jsonPath, mdPath } = await writeBenchmarkReports(report);

  console.log("\n==================================================");
  console.log(`BENCHMARK RESULTS: ${report.passed}/${report.totalCases} PASSED`);
  console.log(`Overall Score: ${report.overallScore}/100 [Status: ${report.overallStatus}]`);
  console.log(`Model Calls Avoided: ${report.efficiency.modelAvoidanceRate} (${report.efficiency.modelCallsAvoided}/${report.efficiency.totalRequests})`);
  console.log(`Deterministic Rate: ${report.efficiency.deterministicRate}`);
  console.log(`Average Latency: ${report.efficiency.avgLatencyMs}ms (P95: ${report.efficiency.p95LatencyMs}ms)`);
  console.log(`Reports generated:`);
  console.log(`- JSON: ${jsonPath}`);
  console.log(`- Markdown: ${mdPath}`);
  console.log("==================================================\n");

  if (report.overallStatus === "CRITICAL_FAILURE") {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error("Benchmark runner failed:", err);
  process.exit(1);
});
