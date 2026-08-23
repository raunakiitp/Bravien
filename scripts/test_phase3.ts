/**
 * Automated verification for Phase 3: AI Workspace & Project System.
 */

import { validateUpload } from "../src/lib/files/validate";
import { chunkDocument } from "../src/lib/rag/chunker";
import { formatRetrievedContext } from "../src/lib/rag/retrieval";
import { buildSystemPromptDetailed } from "../src/lib/ai/prompts";
import { prisma } from "../src/lib/db/prisma";
import { createProject, getProject, updateProject, deleteProject, listProjects } from "../src/lib/db/projects";
import { createConversation, listConversations } from "../src/lib/db/conversations";
import { searchConversations } from "../src/lib/search/conversations";
import { createMemory, listMemories } from "../src/lib/memory/service";

async function runTests() {
  console.log("=== BRAVIEN PHASE 3 VERIFICATION SUITE ===\n");
  let passed = 0;
  let failed = 0;

  function assert(cond: boolean, name: string) {
    if (cond) {
      console.log(`✅ PASS: ${name}`);
      passed++;
    } else {
      console.error(`❌ FAIL: ${name}`);
      failed++;
    }
  }

  // 1. File Upload Validation
  console.log("--- 1. File Validation & Formats ---");
  const pdfVal = validateUpload({ filename: "research.pdf", mimeType: "application/pdf", size: 1024 * 100 });
  assert(pdfVal.ok === true, "PDF upload is allowed");

  const docxVal = validateUpload({ filename: "notes.docx", mimeType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document", size: 1024 * 100 });
  assert(docxVal.ok === true, "DOCX upload is allowed");

  const exeVal = validateUpload({ filename: "malware.exe", mimeType: "application/x-msdownload", size: 100 });
  assert(exeVal.ok === false, "EXE upload is rejected");

  // 2. Document Chunking
  console.log("\n--- 2. Deterministic Boundary Chunking ---");
  const longText = Array.from({ length: 30 }, (_, i) => `Paragraph ${i + 1}: Deep learning architectures in Bravien operate with local inference on GPU hardware.`).join("\n\n");
  const chunks = chunkDocument(longText, { chunkSize: 200, chunkOverlap: 40 });
  assert(chunks.length > 1, `Document chunked into ${chunks.length} chunks`);
  assert(chunks[0].chunkIndex === 0 && chunks[0].content.includes("Paragraph 1"), "Chunk 0 starts at beginning");
  assert(chunks.every(c => c.tokenCount > 0 && c.content.length > 0), "All chunks have positive length and tokens");

  // 3. System Prompt & Project Context
  console.log("\n--- 3. System Prompt Construction ---");
  const promptRes = buildSystemPromptDetailed({
    projectInstructions: "Always respond in bullet points and cite references.",
    memories: ["User prefers TypeScript."],
    projectDocumentsContext: "[Source Document: ml.pdf (Section 1)]\nModel is Qwen2.5-0.5B.",
    contextWindow: 32768,
  });
  assert(promptRes.prompt.includes("Always respond in bullet points"), "Prompt contains project instructions");
  assert(promptRes.prompt.includes("User prefers TypeScript"), "Prompt contains memories");
  assert(promptRes.prompt.includes("Source Document: ml.pdf"), "Prompt contains retrieved document context");

  // 4. Database & Project Isolation
  console.log("\n--- 4. Database CRUD & Workspace Isolation ---");
  const testUser = await prisma.user.findFirst();
  if (!testUser) {
    console.error("No user found in database. Please seed the DB.");
    process.exit(1);
  }
  const userId = testUser.id;

  // Create Project
  const project = await createProject({
    userId,
    name: "Phase 3 Test Workspace",
    description: "Automated test project",
    instructions: "Strict test instructions",
  });
  assert(Boolean(project.id) && project.name === "Phase 3 Test Workspace", "Project created successfully");

  // Get & Update Project
  const fetched = await getProject(userId, project.id);
  assert(fetched?.instructions === "Strict test instructions", "Project instructions retrieved");

  const updated = await updateProject(userId, project.id, {
    instructions: "Updated instructions v2",
  });
  assert(updated?.instructions === "Updated instructions v2", "Project instructions updated");

  // Create Project-scoped Conversation
  const conv = await createConversation({
    userId,
    projectId: project.id,
    title: "Quantum Mechanics Discussion",
    model: "bravien-local",
  });
  assert(conv.projectId === project.id, "Conversation associated with project");

  // Create Project-scoped Memory
  const mem = await createMemory({
    userId,
    projectId: project.id,
    type: "PROJECT",
    content: "Project target deadline is next month.",
  });
  assert(mem.projectId === project.id, "Memory scoped to project");

  // List project memories
  const projectMems = await listMemories(userId, { projectId: project.id });
  assert(projectMems.some(m => m.id === mem.id), "Project memory listed under project filter");

  // Search Conversations
  const searchHits = await searchConversations(userId, "Quantum", { limit: 10 });
  assert(searchHits.some(h => h.conversationId === conv.id), "Conversation search found title match");

  // List Projects with counts
  const projectsList = await listProjects(userId);
  const foundProj = projectsList.find(p => p.id === project.id);
  assert(foundProj !== undefined && (foundProj.conversationCount ?? 0) >= 1, "Project listed with conversation count");

  // Clean up
  await deleteProject(userId, project.id);
  const deletedCheck = await getProject(userId, project.id);
  assert(deletedCheck === null, "Project and cascading items deleted successfully");

  console.log(`\n==========================================`);
  console.log(`RESULTS: ${passed} passed, ${failed} failed.`);
  console.log(`==========================================\n`);

  if (failed > 0) process.exit(1);
}

runTests().catch((err) => {
  console.error("Test error:", err);
  process.exit(1);
});
