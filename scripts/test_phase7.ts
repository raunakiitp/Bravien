/**
 * Automated verification suite for Phase 7: Bravien Autonomous Workspace & Task Execution System.
 */

import { prisma } from "../src/lib/db/prisma";
import {
  createTask,
  getTaskById,
  listTasks,
  updateTask,
  deleteTask,
  logActivity,
  listRecentActivities,
} from "../src/lib/tasks/service";
import { executeTask, MAX_ALLOWED_TASK_STEPS } from "../src/lib/tasks/executor";
import { createProject } from "../src/lib/db/projects";

async function runPhase7Tests() {
  console.log("==================================================");
  console.log("BRAVIEN PHASE 7 AUTOMATED VERIFICATION SUITE");
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

  try {
    // Ensure test users exist
    const userA = await prisma.user.upsert({
      where: { email: "phase7_userA@example.com" },
      update: {},
      create: { email: "phase7_userA@example.com", name: "User A" },
    });

    const userB = await prisma.user.upsert({
      where: { email: "phase7_userB@example.com" },
      update: {},
      create: { email: "phase7_userB@example.com", name: "User B" },
    });

    // Clean previous test data
    await prisma.task.deleteMany({ where: { userId: { in: [userA.id, userB.id] } } });
    await prisma.activityLog.deleteMany({ where: { userId: { in: [userA.id, userB.id] } } });
    await prisma.project.deleteMany({ where: { userId: { in: [userA.id, userB.id] } } });

    // Create test project for user A
    const projectA = await createProject({
      userId: userA.id,
      name: "Autonomous Workspace Test",
      description: "Testing project-scoped autonomous tasks",
      instructions: "Strictly focus on performance and correctness.",
    });

    // --- PART 1: Task Creation & Validation ---
    console.log("--- Part 1: Task Creation & Validation ---");
    const task1 = await createTask(userA.id, {
      title: "Research Next.js 15 Server Actions",
      description: "Find latest capabilities, streaming optimizations, and breaking changes.",
      type: "RESEARCH",
      priority: "HIGH",
      projectId: projectA.id,
    });

    assert(Boolean(task1.id && task1.title === "Research Next.js 15 Server Actions"), "Task created successfully with title");
    assert(task1.status === "PENDING", "Initial task status is PENDING");
    assert(task1.priority === "HIGH", "Task priority stored as HIGH");
    assert(task1.projectId === projectA.id, "Task correctly scoped to project A");

    // Verify project ownership check during creation
    let crossProjectFailed = false;
    try {
      await createTask(userB.id, {
        title: "Malicious Task",
        projectId: projectA.id, // User B trying to attach User A's project
      });
    } catch {
      crossProjectFailed = true;
    }
    assert(crossProjectFailed, "Security: User B cannot attach task to User A's project");

    // --- PART 2: Task CRUD & Filtering ---
    console.log("\n--- Part 2: Task CRUD & Ownership ---");
    const fetchedTask = await getTaskById(userA.id, task1.id);
    assert(fetchedTask.id === task1.id && fetchedTask.project?.id === projectA.id, "getTaskById returned task with project");

    // Security: User B cannot access User A's task
    let idorFetchFailed = false;
    try {
      await getTaskById(userB.id, task1.id);
    } catch {
      idorFetchFailed = true;
    }
    assert(idorFetchFailed, "Security: User B cannot fetch User A's task (IDOR protection)");

    // Update task
    const updatedTask = await updateTask(userA.id, task1.id, {
      priority: "URGENT",
      description: "Updated description for test.",
    });
    assert(updatedTask.priority === "URGENT" && updatedTask.description === "Updated description for test.", "Task updated successfully");

    // List tasks with filter
    const listRes = await listTasks(userA.id, { type: "RESEARCH" });
    assert(listRes.total === 1 && listRes.tasks[0].id === task1.id, "listTasks filtered correctly by type");

    // --- PART 3: Autonomous Task Execution & Step Tracking ---
    console.log("\n--- Part 3: Task Execution & Step Tracking ---");
    const execResult = await executeTask(userA.id, task1.id, { timeoutMs: 10000 });

    assert(execResult.status === "COMPLETED", "Task completed execution successfully");
    assert(execResult.stepsCount <= MAX_ALLOWED_TASK_STEPS, `Task steps bounded under max ${MAX_ALLOWED_TASK_STEPS}`);
    assert(execResult.successfulSteps > 0, "Task recorded successful execution steps");
    assert(Boolean(execResult.resultSummary && execResult.resultSummary.includes("completed")), "Task produced formatted result summary");

    // Check persistent DB state after execution
    const postExecTask = await getTaskById(userA.id, task1.id);
    assert(postExecTask.status === "COMPLETED", "Task status in DB is COMPLETED");
    assert(postExecTask.steps.length > 0, "TaskStep rows persisted in DB");
    assert(postExecTask.executions.length === 1, "TaskExecution record persisted in DB");
    assert(postExecTask.executions[0].status === "COMPLETED", "TaskExecution status in DB is COMPLETED");

    // --- PART 4: General & Analysis Task Execution ---
    console.log("\n--- Part 4: Analysis & Calculation Task Execution ---");
    const mathTask = await createTask(userA.id, {
      title: "Calculate GPU Memory Bandwidth: calculate 128 * 16 / 4",
      type: "ANALYSIS",
      priority: "NORMAL",
    });

    const mathExec = await executeTask(userA.id, mathTask.id);
    assert(mathExec.status === "COMPLETED", "Math analysis task executed successfully");
    assert(mathExec.resultSummary.includes("512"), "Math analysis task used calculator tool and calculated result");

    // --- PART 5: Execution Safety & Failure Handling ---
    console.log("\n--- Part 5: Execution Safety & Failure Recovery ---");
    let idorExecFailed = false;
    try {
      await executeTask(userB.id, task1.id); // User B executing User A's task
    } catch {
      idorExecFailed = true;
    }
    assert(idorExecFailed, "Security: User B cannot trigger execution of User A's task");

    // --- PART 6: Activity Logging & Secret Sanitization ---
    console.log("\n--- Part 6: Activity Logging & Secret Sanitization ---");
    await logActivity(userA.id, {
      eventType: "SECURITY_TEST",
      description: "Testing credential sanitization",
      metadata: {
        userToken: "secret_token_12345",
        passwordHash: "super_secret_hash",
        safeInfo: "safe_public_data",
      },
    });

    const userAActivities = await listRecentActivities(userA.id, { limit: 10 });
    assert(userAActivities.length >= 3, `Activity log captured ${userAActivities.length} user events`);

    const secLog = userAActivities.find((a) => a.eventType === "SECURITY_TEST");
    const meta = (secLog?.metadata as Record<string, unknown>) || {};
    assert(meta.userToken === "[REDACTED]", "Activity log redacted userToken");
    assert(meta.passwordHash === "[REDACTED]", "Activity log redacted passwordHash");
    assert(meta.safeInfo === "safe_public_data", "Activity log preserved safe fields");

    // --- PART 7: Task Deletion & Cascading Clean-up ---
    console.log("\n--- Part 7: Task Deletion & Cascades ---");
    const deleteRes = await deleteTask(userA.id, mathTask.id);
    assert(deleteRes.success, "Task deleted successfully");

    const remainingSteps = await prisma.taskStep.count({ where: { taskId: mathTask.id } });
    const remainingExecs = await prisma.taskExecution.count({ where: { taskId: mathTask.id } });
    assert(remainingSteps === 0 && remainingExecs === 0, "Cascaded task steps and executions deleted cleanly");

    // Clean-up test data
    await prisma.task.deleteMany({ where: { userId: { in: [userA.id, userB.id] } } });
    await prisma.activityLog.deleteMany({ where: { userId: { in: [userA.id, userB.id] } } });
    await prisma.project.deleteMany({ where: { userId: { in: [userA.id, userB.id] } } });
    await prisma.user.deleteMany({ where: { id: { in: [userA.id, userB.id] } } });
  } catch (err: any) {
    if (err.message?.includes("Can't reach database server") || err.name === "PrismaClientInitializationError") {
      console.log("⚠️ Database offline (localhost:5432) - skipping live DB task tests");
    } else {
      throw err;
    }
  }

  console.log("\n==================================================");
  console.log(`RESULTS: ${passed} passed, ${failed} failed.`);
  console.log("==================================================\n");

  if (failed > 0) process.exit(1);
}

runPhase7Tests().catch((err) => {
  console.error("Phase 7 test failure:", err);
  process.exit(1);
});
