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

  // Setup test users
  const userA = await prisma.user.findFirst();
  if (!userA) {
    console.error("Database user not found. Ensure DB is seeded.");
    process.exit(1);
  }
  const userIdA = userA.id;

  // Create a second test user for authorization / isolation testing
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
  const userIdB = userB.id;

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

  // --- PART 2: Memory Intelligence & Duplicate Prevention ---
  console.log("\n--- Part 2: Memory Intelligence & Relevance Ranking ---");
  const mem1 = await createMemory({
    userId: userIdA,
    type: "PREFERENCE",
    content: "Prefers Python for backend data processing and Next.js for frontend.",
  });

  // Duplicate prevention test
  const dupMem = await createMemory({
    userId: userIdA,
    type: "PREFERENCE",
    content: "Prefers Python for backend data processing and Next.js for frontend.",
  });
  assert(dupMem.id === mem1.id, "Duplicate memory creation prevented (returned existing ID)");

  const mem2 = await createMemory({
    userId: userIdA,
    type: "FACT",
    content: "Working on Bravien AI assistant project.",
  });

  // Relevance ranking test
  const relevantMems = await getRelevantMemoriesForQuery(userIdA, "Which backend language do I prefer for data processing?");
  assert(relevantMems.length > 0 && relevantMems[0].includes("Python"), "Relevance ranking placed Python preference first");

  // User B cannot access User A's memory
  const crossUserMem = await getMemory(userIdB, mem1.id);
  assert(crossUserMem === null, "Security: User B cannot access User A's memory (IDOR protection)");

  const crossUserUpdate = await updateMemory(userIdB, mem1.id, { content: "Hacked content" });
  assert(crossUserUpdate === null, "Security: User B cannot update User A's memory");

  const crossUserDelete = await deleteMemory(userIdB, mem1.id);
  assert(crossUserDelete === false, "Security: User B cannot delete User A's memory");

  // --- PART 3: Project Isolation & Security ---
  console.log("\n--- Part 3: Project Workspace Isolation & Security ---");
  const projectA = await createProject({
    userId: userIdA,
    name: "User A Confidential Project",
    instructions: "Confidential proprietary instructions for A.",
  });

  const projectBCheck = await getProject(userIdB, projectA.id);
  assert(projectBCheck === null, "Security: User B cannot access User A's project (IDOR protection)");

  const projectBDelete = await deleteProject(userIdB, projectA.id);
  assert(projectBDelete === false, "Security: User B cannot delete User A's project");

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

  // --- PART 5: RAG Quality, Phrase Match & Chunk Deduplication ---
  console.log("\n--- Part 5: RAG Retrieval & Phrase Match Scoring ---");
  const sampleDoc = "Bravien architecture incorporates local GPU inference via Qwen2.5-0.5B-Instruct running on RTX 4050. It persists state in PostgreSQL and uses Next.js.";
  const docChunks = chunkDocument(sampleDoc);

  const attA = await prisma.attachment.create({
    data: {
      userId: userIdA,
      projectId: projectA.id,
      filename: "architecture_specs.md",
      mimeType: "text/markdown",
      size: sampleDoc.length,
      storagePath: "test_phase5_storage_key",
      extractedText: sampleDoc,
      status: "READY",
      documentChunks: {
        create: docChunks.map((c) => ({
          projectId: projectA.id,
          chunkIndex: c.chunkIndex,
          content: c.content,
          tokenCount: c.tokenCount,
        })),
      },
    },
  });

  const retrieved = await retrieveRelevantContext({
    userId: userIdA,
    projectId: projectA.id,
    query: "local GPU inference via Qwen2.5-0.5B-Instruct",
  });
  assert(retrieved.length > 0, "RAG retrieved matching chunks");
  assert(retrieved[0].score > 10, "Exact phrase match received significant scoring boost");

  // Security: User B cannot retrieve User A's project document chunks
  const crossUserRAG = await retrieveRelevantContext({
    userId: userIdB,
    projectId: projectA.id,
    query: "local GPU inference",
  });
  assert(crossUserRAG.length === 0, "Security: User B cannot retrieve User A's project document chunks");

  // --- PART 6: Conversation Persistence & Clean Regeneration ---
  console.log("\n--- Part 6: Conversation Persistence & Regeneration ---");
  const conv = await createConversation({
    userId: userIdA,
    title: "Phase 5 Regeneration Test",
    model: "Qwen/Qwen2.5-0.5B-Instruct",
  });

  await appendMessage({
    conversationId: conv.id,
    role: "user",
    content: "Explain transformers",
  });

  const partialAssistant = await appendMessage({
    conversationId: conv.id,
    role: "assistant",
    content: "Transformers are attention-based...",
  });

  const msgsBefore = await getConversationMessages(userIdA, conv.id);
  assert(msgsBefore?.length === 2, "Conversation has 2 messages before regeneration");

  // Delete last assistant message for clean regeneration
  const regenCleaned = await deleteLastAssistantMessage(userIdA, conv.id);
  assert(regenCleaned === true, "deleteLastAssistantMessage succeeded");

  const msgsAfter = await getConversationMessages(userIdA, conv.id);
  assert(msgsAfter?.length === 1 && msgsAfter[0].role === "user", "Trailing assistant message removed cleanly for regeneration");

  // Cleanup test resources
  await deleteConversation(userIdA, conv.id);
  await prisma.attachment.delete({ where: { id: attA.id } });
  await deleteProject(userIdA, projectA.id);
  await deleteMemory(userIdA, mem1.id);
  await deleteMemory(userIdA, mem2.id);
  await prisma.user.delete({ where: { id: userIdB } });

  console.log("\n==================================================");
  console.log(`RESULTS: ${passed} passed, ${failed} failed.`);
  console.log("==================================================\n");

  if (failed > 0) process.exit(1);
}

runPhase5Tests().catch((err) => {
  console.error("Phase 5 test failure:", err);
  process.exit(1);
});
