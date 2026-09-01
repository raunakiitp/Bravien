/**
 * Automated verification suite for Phase 5: Bravien Production Intelligence & Security.
 */

import { runToolLoop } from "../src/lib/ai/tools/loop";
import { calculatorTool } from "../src/lib/ai/tools/calculator";
import { timeTool } from "../src/lib/ai/tools/time";
import {
  createMemory,
  getMemory,
  updateMemory,
  deleteMemory,
  getRelevantMemoriesForQuery,
  listMemories,
} from "../src/lib/memory/service";
import { sanitizeFilename, validateUpload } from "../src/lib/files/validate";
import { retrieveRelevantContext } from "../src/lib/rag/retrieval";
import { chunkDocument } from "../src/lib/rag/chunker";
import {
  createConversation,
  appendMessage,
  getConversationMessages,
  deleteLastAssistantMessage,
  deleteConversation,
} from "../src/lib/db/conversations";
import { createProject, deleteProject, getProject } from "../src/lib/db/projects";
import { prisma } from "../src/lib/db/prisma";

async function runPhase5Tests() {
  console.log("==================================================");
  console.log("BRAVIEN PHASE 5 AUTOMATED VERIFICATION SUITE");
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

  let userIdA = "user_a_fallback";
  let userIdB = "user_b_fallback";

  // Setup test users if DB available
  try {
    const userA = await prisma.user.findFirst();
    if (userA) {
      userIdA = userA.id;
      let userB = await prisma.user.findUnique({ where: { email: "user_b_test@bravien.local" } });
      if (!userB) {
        userB = await prisma.user.create({
          data: {
            email: "user_b_test@bravien.local",
            name: "Test User B",
            passwordHash: "test_hash_phase5",
          },
        });
      }
      userIdB = userB.id;
    }
  } catch {
    // offline
  }

  // --- PART 1: Tool Execution Loop & Bounded Protection ---
  console.log("--- Part 1: Tool Execution Loop & Bounded Protection ---");
  const toolLoopResult = await runToolLoop(
    "What is the current date and time and calculate 45 * 20?",
    { userId: userIdA },
    { maxIterations: 3, timeoutMs: 5000 },
  );

  assert(toolLoopResult.executedTools.length >= 1, "Tool loop detected and executed query tools");
  assert(toolLoopResult.combinedFormattedOutput.length > 0, "Tool loop produced formatted combined results");

  // Iteration limit enforcement
  const cappedLoop = await runToolLoop(
    "What is the date and calculate 10 + 10",
    { userId: userIdA },
    { maxIterations: 1 },
  );
  assert(cappedLoop.iterationCount <= 1, "Tool loop strictly enforced maxIterations cap");

  // Calculator error tolerance without crashing
  const badCalc = await calculatorTool.execute({ expression: "10 / (5 - 5)" }, { userId: userIdA });
  assert(!badCalc.success && Boolean(badCalc.error?.includes("Division by zero")), "Calculator safely caught division by zero");

  const invalidSyntaxCalc = await calculatorTool.execute({ expression: "25 + * 4" }, { userId: userIdA });
  assert(!invalidSyntaxCalc.success, "Calculator safely caught syntax error without crashing");

  // --- PART 2 & 3: Memory & Project Isolation (DB) ---
  console.log("\n--- Part 2 & 3: Memory Intelligence & Project Isolation ---");
  try {
    const mem1 = await createMemory({
      userId: userIdA,
      type: "PREFERENCE",
      content: "Prefers Python for backend data processing and Next.js for frontend.",
    });

    const dupMem = await createMemory({
      userId: userIdA,
      type: "PREFERENCE",
      content: "Prefers Python for backend data processing and Next.js for frontend.",
    });
    assert(dupMem.id === mem1.id, "Duplicate memory creation prevented (returned existing ID)");

    const relevantMems = await getRelevantMemoriesForQuery(userIdA, "Which backend language do I prefer for data processing?");
    assert(relevantMems.length > 0 && relevantMems[0].includes("Python"), "Relevance ranking placed Python preference first");

    const crossUserMem = await getMemory(userIdB, mem1.id);
    assert(crossUserMem === null, "Security: User B cannot access User A's memory (IDOR protection)");

    const projectA = await createProject({
      userId: userIdA,
      name: "User A Confidential Project",
      instructions: "Confidential proprietary instructions for A.",
    });

    const projectBCheck = await getProject(userIdB, projectA.id);
    assert(projectBCheck === null, "Security: User B cannot access User A's project (IDOR protection)");

    await deleteProject(userIdA, projectA.id);
    await deleteMemory(userIdA, mem1.id);
  } catch (err: any) {
    if (err.message?.includes("Can't reach database server") || err.name === "PrismaClientInitializationError") {
      console.log("⚠️ Database offline (localhost:5432) - skipping live DB integration checks");
    } else {
      throw err;
    }
  }

  // --- PART 4: Document Ingestion Hardening & Path Traversal ---
  console.log("\n--- Part 4: File Sanitization & Upload Hardening ---");
  const sanitized1 = sanitizeFilename("../../etc/passwd.txt");
  assert(sanitized1 === "passwd.txt" && !sanitized1.includes(".."), "Path traversal sequence ../ stripped");

  const sanitized2 = sanitizeFilename("..\\..\\windows\\system32\\secret.md");
  assert(sanitized2 === "secret.md" && !sanitized2.includes("\\"), "Windows path traversal sequence ..\\ stripped");

  const sanitizedNull = sanitizeFilename("exploit\0file.json");
  assert(!sanitizedNull.includes("\0"), "Null bytes removed from filename");

  const validTxt = validateUpload({ filename: "notes.txt", mimeType: "text/plain", size: 1024 });
  assert(validTxt.ok, "Valid text upload accepted");

  const emptyFile = validateUpload({ filename: "empty.txt", mimeType: "text/plain", size: 0 });
  assert(!emptyFile.ok && emptyFile.code === "EMPTY_FILE", "Empty 0-byte file rejected");

  const invalidExt = validateUpload({ filename: "malware.exe", mimeType: "application/x-msdownload", size: 1024 });
  assert(!invalidExt.ok, "Unsupported file extension rejected");

  console.log("\n==================================================");
  console.log(`RESULTS: ${passed} passed, ${failed} failed.`);
  console.log("==================================================\n");

  if (failed > 0) process.exit(1);
}

runPhase5Tests().catch((err) => {
  console.error("Phase 5 test failure:", err);
  process.exit(1);
});
