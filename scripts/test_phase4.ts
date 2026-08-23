/**
 * Automated verification suite for Phase 4: Bravien Intelligence & Tool System.
 */

import { BRAVIEN_IDENTITY, formatAssistantIdentity } from "../src/lib/ai/identity";
import {
  createMemory,
  getMemory,
  updateMemory,
  deleteMemory,
  clearMemories,
  searchMemories,
  listMemories,
} from "../src/lib/memory/service";
import { executeTool, listTools, detectAndExecuteTools, BUILTIN_TOOLS } from "../src/lib/ai/tools/registry";
import { calculatorTool } from "../src/lib/ai/tools/calculator";
import { timeTool } from "../src/lib/ai/tools/time";
import { buildContext } from "../src/lib/ai/context";
import { summarizeMessagesLocally } from "../src/lib/ai/summarizer";
import { defaultVisionProvider } from "../src/lib/vision/provider";
import { isSpeechRecognitionSupported, isSpeechSynthesisSupported } from "../src/lib/voice/speech";
import { prisma } from "../src/lib/db/prisma";
import { createProject, deleteProject } from "../src/lib/db/projects";
import type { AIMessage } from "../src/types";

async function runPhase4Tests() {
  console.log("==================================================");
  console.log("BRAVIEN PHASE 4 AUTOMATED VERIFICATION SUITE");
  console.log("==================================================\n");

  let passed = 0;
  let failed = 0;

  function assert(condition: boolean, name: string) {
    if (condition) {
      console.log(`✅ PASS: ${name}`);
      passed++;
    } else {
      console.error(`❌ FAIL: ${name}`);
      failed++;
    }
  }

  // PART 1: Bravien Identity
  console.log("--- Part 1: Bravien Identity ---");
  assert(BRAVIEN_IDENTITY.name === "Bravien", "Identity name is strictly Bravien");
  assert(BRAVIEN_IDENTITY.systemCore.includes("Always identify yourself as Bravien"), "System core enforces Bravien identity");
  const identityFormat = formatAssistantIdentity("Qwen/Qwen2.5-0.5B-Instruct");
  assert(identityFormat.assistant === "Bravien" && identityFormat.engine === "Qwen/Qwen2.5-0.5B-Instruct", "Assistant is Bravien and engine is separated");

  // PART 2: Memory Layer
  console.log("\n--- Part 2: Memory System & Categories ---");
  const testUser = await prisma.user.findFirst();
  if (!testUser) {
    console.error("Database user not found. Ensure DB is seeded.");
    process.exit(1);
  }
  const userId = testUser.id;

  // Create memories across categories
  const memPref = await createMemory({
    userId,
    type: "PREFERENCE",
    content: "User prefers dark mode and concise TypeScript answers.",
  });
  assert(memPref.type === "PREFERENCE" && Boolean(memPref.id), "Created PREFERENCE memory");

  const memFact = await createMemory({
    userId,
    type: "FACT",
    content: "User is building Bravien local AI assistant on RTX 4050 GPU.",
  });
  assert(memFact.type === "FACT", "Created FACT memory");

  const memWorkflow = await createMemory({
    userId,
    type: "WORKFLOW",
    content: "Always run npm run typecheck before deployment.",
  });
  assert(memWorkflow.type === "WORKFLOW", "Created WORKFLOW memory");

  // Search memories
  const foundMem = await searchMemories(userId, "dark mode");
  assert(foundMem.some((m) => m.id === memPref.id), "searchMemories found preference match");

  // Update memory
  const updatedMem = await updateMemory(userId, memPref.id, {
    content: "User prefers ultra dark mode with JetBrains Mono.",
  });
  assert(Boolean(updatedMem?.content.includes("ultra dark")), "updateMemory succeeded");

  // PART 3 & 4: Modular Tools & Execution
  console.log("\n--- Part 3 & 4: Tool Registry & Deterministic Execution ---");
  const toolList = listTools();
  assert(toolList.length >= 6, `Registered ${toolList.length} built-in tools`);
  assert(Boolean(BUILTIN_TOOLS["calculator"]), "Calculator tool registered");
  assert(Boolean(BUILTIN_TOOLS["get_current_time"]), "Time tool registered");
  assert(Boolean(BUILTIN_TOOLS["search_memories"]), "Memory search tool registered");
  assert(Boolean(BUILTIN_TOOLS["search_documents"]), "Document search tool registered");

  // Calculator Math Tests
  const calc1 = await calculatorTool.execute({ expression: "25 * 4 + 10 / 2" }, { userId });
  assert(calc1.success && (calc1.data as { result: number }).result === 105, "Calculator standard precedence: 25*4+10/2 = 105");

  const calc2 = await calculatorTool.execute({ expression: "sqrt(144) + 2^3" }, { userId });
  assert(calc2.success && (calc2.data as { result: number }).result === 20, "Calculator functions: sqrt(144) + 2^3 = 20");

  // Time Tool Test
  const timeRes = await timeTool.execute({}, { userId });
  assert(timeRes.success && Boolean((timeRes.data as { date: string }).date), "Time tool returned current date & time");

  // Deterministic tool detection
  const detectedTime = await detectAndExecuteTools("What is today's date and time?", { userId });
  assert(detectedTime.length > 0 && detectedTime[0].success, "Auto-detected time request");

  const detectedMath = await detectAndExecuteTools("Calculate 125 * 8", { userId });
  assert(detectedMath.length > 0 && (detectedMath[0].data as { result: number }).result === 1000, "Auto-detected math calculation 125 * 8 = 1000");

  // PART 5 & 6: Context Management & Summarization
  console.log("\n--- Part 5 & 6: Context Builder & Summarization ---");
  const testMessages: AIMessage[] = [
    { role: "user", content: "Hello Bravien" },
    { role: "assistant", content: "Hello! How can I help you today?" },
    { role: "user", content: "Write a python script to benchmark GPU throughput." },
  ];

  const builtCtx = buildContext({
    messages: testMessages,
    contextWindow: 4096,
    projectInstructions: "Project scope: PyTorch kernels",
    toolResultsFormatted: "Calculation Result: 125 * 8 = 1000",
    isCodingMode: true,
  });

  assert(builtCtx.systemPrompt.includes("Bravien"), "Context system prompt contains Bravien identity");
  assert(builtCtx.systemPrompt.includes("PyTorch kernels"), "Context system prompt contains project instructions");
  assert(builtCtx.systemPrompt.includes("Calculation Result: 125 * 8 = 1000"), "Context system prompt contains tool results");
  assert(builtCtx.systemPrompt.includes("When writing code"), "Context system prompt contains coding mode guidelines");

  // Summarization test
  const summaryText = summarizeMessagesLocally(testMessages);
  assert(summaryText.includes("Prior Conversation Context Summary"), "Summarizer generated structured history summary");

  // PART 7 & 8: Vision & Voice Abstractions
  console.log("\n--- Part 7 & 8: Voice & Vision Architecture Fallbacks ---");
  const visionRes = await defaultVisionProvider.analyzeImage({
    imageBuffer: Buffer.from("fake-png-data"),
    mimeType: "image/png",
    filename: "chart.png",
  });
  assert(visionRes.supported === false && visionRes.error === "VISION_UNAVAILABLE", "Vision gracefully reports text-only runtime status");

  // Voice capability check (server environment test)
  const isSpeechRec = isSpeechRecognitionSupported();
  assert(typeof isSpeechRec === "boolean", "Speech recognition capability is safely queryable in any runtime");

  // PART 9: Project Isolation & Security
  console.log("\n--- Part 9: Project Security & Memory Clean-up ---");
  const proj = await createProject({
    userId,
    name: "Phase 4 Security Test Project",
    instructions: "Strict test instructions",
  });

  const projMem = await createMemory({
    userId,
    projectId: proj.id,
    type: "PROJECT",
    content: "Project confidential key: xyz987",
  });

  // Verify memory scoping
  const scopedMems = await listMemories(userId, { projectId: proj.id });
  assert(scopedMems.some((m) => m.id === projMem.id), "Project memory list is properly scoped");

  // Clean up
  await deleteMemory(userId, memPref.id);
  await deleteMemory(userId, memFact.id);
  await deleteMemory(userId, memWorkflow.id);
  await deleteProject(userId, proj.id);

  console.log("\n==================================================");
  console.log(`RESULTS: ${passed} passed, ${failed} failed.`);
  console.log("==================================================\n");

  if (failed > 0) process.exit(1);
}

runPhase4Tests().catch((err) => {
  console.error("Phase 4 test error:", err);
  process.exit(1);
});
