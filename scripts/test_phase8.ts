/**
 * BRAVIEN PHASE 8 AUTOMATED VERIFICATION SUITE
 *
 * Tests the Autonomous Agent Memory + Planning + Execution Intelligence system:
 * - AgentState creation, ownership, transitions, bounds, deduplication
 * - Action proposals, risk classification, confirmation approval/rejection, cross-user isolation
 * - Memory promotion with strict secret rejection
 * - Task checkpointing, resumption, transient failure retries
 * - Evidence continuity, compact context summarization, and activity logging
 */

import { prisma } from "../src/lib/db/prisma";
import {
  getOrCreateActiveAgentState,
  getAgentStateById,
  getRelevantAgentState,
  updateAgentState,
  transitionState,
  addConstraint,
  recordDecision,
  recordCompletedAction,
  recordEvidenceRef,
  summarizeAgentStateForContext,
  isValidStateTransition,
} from "../src/lib/ai/agent-state";
import {
  createActionProposal,
  approveActionProposal,
  rejectActionProposal,
  classifyActionRisk,
  listPendingActionProposals,
} from "../src/lib/ai/action-proposal";
import {
  promoteStateToPersistentMemory,
  containsSecretPatterns,
} from "../src/lib/ai/memory-promotion";
import { createTask, getTaskById, listTasks, logActivity } from "../src/lib/tasks/service";
import { executeTask, resumeTask } from "../src/lib/tasks/executor";
import { buildContext } from "../src/lib/ai/context";

let passed = 0;
let failed = 0;

function assert(condition: boolean, message: string) {
  if (condition) {
    console.log(`✅ PASS: ${message}`);
    passed++;
  } else {
    console.error(`❌ FAIL: ${message}`);
    failed++;
  }
}

async function run() {
  console.log("==================================================");
  console.log("BRAVIEN PHASE 8 AUTOMATED VERIFICATION SUITE");
  console.log("==================================================\n");

  const timestamp = Date.now();
  const testUserAEmail = `phase8_userA_${timestamp}@bravien.local`;
  const testUserBEmail = `phase8_userB_${timestamp}@bravien.local`;

  // Create test users
  const userA = await prisma.user.create({
    data: { email: testUserAEmail, name: "Phase 8 User A" },
  });
  const userB = await prisma.user.create({
    data: { email: testUserBEmail, name: "Phase 8 User B" },
  });

  const projectA = await prisma.project.create({
    data: { userId: userA.id, name: `Project A ${timestamp}` },
  });

  try {
    // ----------------------------------------------------
    // Part 1: AgentState Creation & Retrieval
    // ----------------------------------------------------
    console.log("--- Part 1: AgentState Creation & Retrieval ---");
    const stateA = await getOrCreateActiveAgentState(userA.id, {
      projectId: projectA.id,
      goal: "Build autonomous workflow engine",
    });

    assert(Boolean(stateA.id), "AgentState created with valid ID");
    assert(stateA.status === "IDLE", "Initial state status is IDLE");
    assert(stateA.goal === "Build autonomous workflow engine", "Goal preserved accurately");

    const fetchedA = await getAgentStateById(userA.id, stateA.id);
    assert(fetchedA.id === stateA.id, "getAgentStateById returned correct state");

    const relevant = await getRelevantAgentState(userA.id, { projectId: projectA.id });
    assert(relevant?.id === stateA.id, "getRelevantAgentState retrieved state for project");

    // Cross-user IDOR check
    let userBAccessBlocked = false;
    try {
      await getAgentStateById(userB.id, stateA.id);
    } catch {
      userBAccessBlocked = true;
    }
    assert(userBAccessBlocked, "Security: User B cannot access User A's AgentState (IDOR blocked)");

    // ----------------------------------------------------
    // Part 2: State Transitions & Validation
    // ----------------------------------------------------
    console.log("\n--- Part 2: State Transitions & Validation ---");
    assert(isValidStateTransition("IDLE", "PLANNING"), "IDLE -> PLANNING is valid");
    assert(isValidStateTransition("PLANNING", "EXECUTING"), "PLANNING -> EXECUTING is valid");
    assert(!isValidStateTransition("IDLE", "COMPLETED"), "IDLE -> COMPLETED is invalid transition");

    const planState = await transitionState(userA.id, stateA.id, "PLANNING");
    assert(planState.status === "PLANNING", "transitionState moved status to PLANNING");

    let invalidTransitionBlocked = false;
    try {
      // Trying an invalid transition from PLANNING -> IDLE directly
      await transitionState(userA.id, stateA.id, "IDLE");
    } catch {
      invalidTransitionBlocked = true;
    }
    assert(invalidTransitionBlocked, "Invalid state transition safely rejected");

    // ----------------------------------------------------
    // Part 3: Constraints, Decisions & Deduplication
    // ----------------------------------------------------
    console.log("\n--- Part 3: Constraints, Decisions & Deduplication ---");
    await addConstraint(userA.id, stateA.id, "Use local GPU inference only");
    await addConstraint(userA.id, stateA.id, "Use local GPU inference only"); // Duplicate
    await addConstraint(userA.id, stateA.id, "Never expose user credentials");

    const withConstraints = await getAgentStateById(userA.id, stateA.id);
    assert(withConstraints.constraints.length === 2, "Constraints deduplicated accurately (2 unique items)");
    assert(withConstraints.constraints.includes("Use local GPU inference only"), "Constraint stored correctly");

    await recordDecision(userA.id, stateA.id, "Selected PostgreSQL for persistent state");
    await recordDecision(userA.id, stateA.id, "Selected PostgreSQL for persistent state"); // Duplicate
    await recordDecision(userA.id, stateA.id, "Implemented Prisma client v6.19.3");

    const withDecisions = await getAgentStateById(userA.id, stateA.id);
    assert(withDecisions.decisions.length === 2, "Decisions deduplicated accurately (2 unique items)");

    await recordCompletedAction(userA.id, stateA.id, "Database schema synchronized");
    const withActions = await getAgentStateById(userA.id, stateA.id);
    assert(withActions.completedActions.length === 1, "Completed action recorded");

    // ----------------------------------------------------
    // Part 4: Evidence Continuity & Deduplication
    // ----------------------------------------------------
    console.log("\n--- Part 4: Evidence Continuity & Deduplication ---");
    await recordEvidenceRef(userA.id, stateA.id, {
      title: "Prisma Docs",
      url: "https://pris.ly/docs",
      sourceType: "WEB",
      snippet: "Prisma ORM documentation",
      timestamp: new Date().toISOString(),
    });
    await recordEvidenceRef(userA.id, stateA.id, {
      title: "Prisma Docs",
      url: "https://pris.ly/docs",
      sourceType: "WEB",
      snippet: "Prisma ORM documentation duplicate",
      timestamp: new Date().toISOString(),
    });

    const stateWithEvidence = await getAgentStateById(userA.id, stateA.id);
    const evidenceArray = stateWithEvidence.evidenceRefs as Array<{ url: string }>;
    assert(Array.isArray(evidenceArray) && evidenceArray.length === 1, "Evidence reference deduplicated by URL");

    // ----------------------------------------------------
    // Part 5: Action Proposals & Confirmation Gate
    // ----------------------------------------------------
    console.log("\n--- Part 5: Action Proposals & Confirmation Gate ---");
    const lowRisk = classifyActionRisk("search_web");
    assert(lowRisk.riskLevel === "LOW" && !lowRisk.requiresConfirmation, "search_web classified as LOW risk");

    const highRisk = classifyActionRisk("delete_database_records");
    assert(highRisk.riskLevel === "HIGH" && highRisk.requiresConfirmation, "delete_database_records classified as HIGH risk requiring confirmation");

    const proposal = await createActionProposal(userA.id, {
      actionType: "delete_project_assets",
      description: "Remove obsolete project embeddings",
      agentStateId: stateA.id,
    });

    assert(proposal.status === "PENDING", "Action proposal created with PENDING status");
    assert(proposal.requiresConfirmation === true, "High-risk action proposal marked requiring confirmation");

    // Cross-user approval IDOR rejection
    let userBApproveBlocked = false;
    try {
      await approveActionProposal(userB.id, proposal.id);
    } catch {
      userBApproveBlocked = true;
    }
    assert(userBApproveBlocked, "Security: User B cannot approve User A's action proposal");

    const approved = await approveActionProposal(userA.id, proposal.id);
    assert(approved.status === "APPROVED", "User A successfully approved action proposal");

    // Reject flow test
    const proposalToReject = await createActionProposal(userA.id, {
      actionType: "truncate_logs",
      description: "Truncate log table",
    });
    const rejected = await rejectActionProposal(userA.id, proposalToReject.id);
    assert(rejected.status === "REJECTED", "User A successfully rejected action proposal");

    // ----------------------------------------------------
    // Part 6: Memory Promotion & Secret Filtering
    // ----------------------------------------------------
    console.log("\n--- Part 6: Memory Promotion & Secret Filtering ---");
    assert(containsSecretPatterns("My password is secret123"), "Secret detector caught password");
    assert(containsSecretPatterns("api_key=sk-abcdef1234567890abcdef1234567890"), "Secret detector caught API key");
    assert(!containsSecretPatterns("Always use Python 3.11 with strict typing"), "Secret detector permitted normal rule");

    // Add a secret constraint to test rejection
    await addConstraint(userA.id, stateA.id, "password=SuperSecretPassword123!");
    await addConstraint(userA.id, stateA.id, "Always write clean modular code");

    const promoResult = await promoteStateToPersistentMemory(userA.id, stateA.id, {
      projectId: projectA.id,
    });

    assert(promoResult.promotedCount >= 2, `Promoted ${promoResult.promotedCount} safe facts to persistent memory`);
    const promotedContents = promoResult.memories.map((m) => m.content);
    assert(!promotedContents.some((c) => c.includes("SuperSecretPassword123!")), "Security: Password secret was NOT promoted to long-term memory");
    assert(promotedContents.includes("Always write clean modular code"), "Safe rule promoted successfully");

    // ----------------------------------------------------
    // Part 7: Task Execution, Checkpoints & Resumption
    // ----------------------------------------------------
    console.log("\n--- Part 7: Task Execution, Checkpoints & Resumption ---");
    const task = await createTask(userA.id, {
      title: "Calculate annual compounded growth (1000 * 1.08^5)",
      type: "ANALYSIS",
      priority: "HIGH",
      projectId: projectA.id,
    });

    const execResult = await executeTask(userA.id, task.id);
    assert(execResult.status === "COMPLETED", "Task completed execution");
    assert(execResult.successfulSteps > 0, "Task recorded successful step executions");

    // Verify task step checkpointing
    const completedSteps = await prisma.taskStep.findMany({
      where: { taskId: task.id },
      orderBy: { stepNumber: "asc" },
    });
    assert(completedSteps.every((s) => s.status === "COMPLETED"), "All task steps checkpointed as COMPLETED");

    // Test resumeTask on completed task (should reuse existing checkpoints)
    const resumeResult = await resumeTask(userA.id, task.id);
    assert(resumeResult.status === "COMPLETED", "resumeTask completed successfully using checkpoints");

    // ----------------------------------------------------
    // Part 8: Context Summarization for Qwen 0.5B
    // ----------------------------------------------------
    console.log("\n--- Part 8: Context Summarization for Qwen 0.5B ---");
    const summary = summarizeAgentStateForContext({
      goal: stateA.goal,
      status: "EXECUTING",
      currentStep: 2,
      constraints: ["Use local GPU inference only", "Never expose credentials"],
      decisions: ["Selected PostgreSQL for state"],
      completedActions: ["Database schema synchronized"],
    });

    assert(summary.includes("=== ACTIVE AGENT STATE ==="), "Context summary contains header");
    assert(summary.includes("GOAL: Build autonomous workflow engine"), "Context summary contains goal");
    assert(summary.includes("ACTIVE CONSTRAINTS:"), "Context summary contains constraints");
    assert(summary.length < 1500, `Context summary is ultra-compact (${summary.length} chars)`);

    // Verify buildContext accepts agentStateSummary
    const built = buildContext({
      messages: [{ role: "user", content: "What is our current plan?" }],
      agentStateSummary: summary,
    });
    assert(built.systemPrompt.includes("=== ACTIVE AGENT STATE ==="), "buildContext correctly injected agent state summary");

    // ----------------------------------------------------
    // Part 9: Activity Logging
    // ----------------------------------------------------
    console.log("\n--- Part 9: Activity Logging ---");
    const userActivities = await prisma.activityLog.findMany({
      where: { userId: userA.id },
      orderBy: { createdAt: "desc" },
    });
    assert(userActivities.length >= 5, `Activity log captured ${userActivities.length} user events`);
    const eventTypes = userActivities.map((a) => a.eventType);
    assert(eventTypes.includes("AGENT_STATE_CREATED"), "Captured AGENT_STATE_CREATED event");
    assert(eventTypes.includes("PLAN_CHECKPOINTED"), "Captured PLAN_CHECKPOINTED event");
    assert(eventTypes.includes("MEMORY_PROMOTED"), "Captured MEMORY_PROMOTED event");

  } finally {
    // Cleanup test data
    await prisma.activityLog.deleteMany({ where: { userId: { in: [userA.id, userB.id] } } });
    await prisma.actionProposal.deleteMany({ where: { userId: { in: [userA.id, userB.id] } } });
    await prisma.agentState.deleteMany({ where: { userId: { in: [userA.id, userB.id] } } });
    await prisma.taskStep.deleteMany({ where: { task: { userId: { in: [userA.id, userB.id] } } } });
    await prisma.taskExecution.deleteMany({ where: { task: { userId: { in: [userA.id, userB.id] } } } });
    await prisma.task.deleteMany({ where: { userId: { in: [userA.id, userB.id] } } });
    await prisma.memory.deleteMany({ where: { userId: { in: [userA.id, userB.id] } } });
    await prisma.project.deleteMany({ where: { userId: { in: [userA.id, userB.id] } } });
    await prisma.user.deleteMany({ where: { id: { in: [userA.id, userB.id] } } });
  }

  console.log("\n==================================================");
  console.log(`RESULTS: ${passed} passed, ${failed} failed.`);
  console.log("==================================================\n");

  if (failed > 0) {
    process.exit(1);
  }
}

run().catch((err) => {
  console.error("Test execution encountered an error:", err);
  process.exit(1);
});
