"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  CheckCircle2,
  Clock,
  ExternalLink,
  FolderKanban,
  ListTodo,
  Loader2,
  Play,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  XCircle,
  AlertCircle,
  Activity,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { useProjects } from "@/hooks/use-projects";
import { cn } from "@/lib/utils";

interface TaskStepItem {
  id: string;
  stepNumber: number;
  title: string;
  description?: string | null;
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED";
  toolName?: string | null;
  toolResult?: { output?: string; [key: string]: unknown } | null;
  error?: string | null;
}

interface TaskItem {
  id: string;
  title: string;
  description?: string | null;
  type: "RESEARCH" | "ANALYSIS" | "DOCUMENT" | "CODING" | "PROJECT" | "GENERAL";
  status: "PENDING" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED";
  priority: "LOW" | "NORMAL" | "HIGH" | "URGENT";
  resultSummary?: string | null;
  createdAt: string;
  updatedAt: string;
  project?: { id: string; name: string } | null;
  steps?: TaskStepItem[];
  _count?: { steps: number; executions: number };
}

interface ActivityItem {
  id: string;
  eventType: string;
  description: string;
  createdAt: string;
  project?: { name: string } | null;
  task?: { title: string } | null;
}

export default function TasksPage() {
  const [tasks, setTasks] = useState<TaskItem[]>([]);
  const [selectedTask, setSelectedTask] = useState<TaskItem | null>(null);
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [executing, setExecuting] = useState(false);
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<string>("ALL");
  const [isCreating, setIsCreating] = useState(false);
  const [activeTab, setActiveTab] = useState<"tasks" | "activity">("tasks");

  // Create form state
  const [newTitle, setNewTitle] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [newType, setNewType] = useState<string>("GENERAL");
  const [newPriority, setNewPriority] = useState<string>("NORMAL");
  const [newProjectId, setNewProjectId] = useState<string>("");
  const [savingTask, setSavingTask] = useState(false);

  const { projects } = useProjects();

  const fetchTasks = useCallback(async () => {
    try {
      const res = await fetch("/api/tasks");
      if (res.ok) {
        const data = await res.json();
        setTasks(data.tasks || []);
      }
    } catch {
      toast.error("Failed to load tasks");
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchActivities = useCallback(async () => {
    try {
      const res = await fetch("/api/activity?limit=30");
      if (res.ok) {
        const data = await res.json();
        setActivities(data.items || []);
      }
    } catch {
      // ignore
    }
  }, []);

  const loadTaskDetails = useCallback(async (taskId: string) => {
    try {
      const res = await fetch(`/api/tasks/${taskId}`);
      if (res.ok) {
        const data = await res.json();
        setSelectedTask(data.task);
      }
    } catch {
      toast.error("Failed to load task details");
    }
  }, []);

  useEffect(() => {
    void fetchTasks();
    void fetchActivities();
  }, [fetchTasks, fetchActivities]);

  async function handleCreateTask(e: React.FormEvent) {
    e.preventDefault();
    if (!newTitle.trim()) return;

    setSavingTask(true);
    try {
      const res = await fetch("/api/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: newTitle.trim(),
          description: newDesc.trim() || undefined,
          type: newType,
          priority: newPriority,
          projectId: newProjectId || undefined,
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error?.message || "Failed to create task");
      }

      const { task } = await res.json();
      toast.success("Task created successfully");
      setIsCreating(false);
      setNewTitle("");
      setNewDesc("");
      setNewProjectId("");
      await fetchTasks();
      setSelectedTask(task);
      void fetchActivities();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Error creating task");
    } finally {
      setSavingTask(false);
    }
  }

  async function handleExecuteTask(taskId: string) {
    setExecuting(true);
    try {
      const res = await fetch(`/api/tasks/${taskId}/execute`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error?.message || "Task execution failed");
      }

      const { result } = await res.json();
      if (result.status === "COMPLETED") {
        toast.success(`Task completed in ${(result.durationMs / 1000).toFixed(1)}s`);
      } else {
        toast.error(`Task execution ended with status: ${result.status}`);
      }

      await loadTaskDetails(taskId);
      await fetchTasks();
      void fetchActivities();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Execution failed");
    } finally {
      setExecuting(false);
    }
  }

  async function handleDeleteTask(taskId: string) {
    if (!window.confirm("Are you sure you want to delete this task?")) return;

    try {
      const res = await fetch(`/api/tasks/${taskId}`, { method: "DELETE" });
      if (res.ok) {
        toast.success("Task deleted");
        if (selectedTask?.id === taskId) setSelectedTask(null);
        await fetchTasks();
        void fetchActivities();
      }
    } catch {
      toast.error("Failed to delete task");
    }
  }

  const filteredTasks = tasks.filter((t) => {
    const matchesSearch =
      t.title.toLowerCase().includes(search.toLowerCase()) ||
      (t.description && t.description.toLowerCase().includes(search.toLowerCase()));
    const matchesType = typeFilter === "ALL" || t.type === typeFilter;
    return matchesSearch && matchesType;
  });

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-4 py-8 sm:px-6">
      {/* Header */}
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b border-border pb-4">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" render={<Link href="/" />}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-foreground flex items-center gap-2">
              <FolderKanban className="h-5 w-5 text-primary" /> Autonomous Tasks &amp; Workspaces
            </h1>
            <p className="text-xs text-muted-foreground">
              Bounded autonomous execution, web research pipelines, and activity tracking.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex rounded-lg border border-border p-0.5 bg-muted/40">
            <button
              onClick={() => setActiveTab("tasks")}
              className={cn(
                "px-3 py-1 text-xs font-medium rounded-md transition-colors",
                activeTab === "tasks" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
              )}
            >
              Tasks ({tasks.length})
            </button>
            <button
              onClick={() => setActiveTab("activity")}
              className={cn(
                "px-3 py-1 text-xs font-medium rounded-md transition-colors",
                activeTab === "activity" ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
              )}
            >
              Activity Feed
            </button>
          </div>

          <Button size="sm" onClick={() => setIsCreating(true)} className="gap-1">
            <Plus className="h-4 w-4" /> New Task
          </Button>
        </div>
      </header>

      {/* Main Content Area */}
      {activeTab === "activity" ? (
        <div className="rounded-xl border border-border bg-card p-6 shadow-sm">
          <h2 className="text-base font-semibold mb-4 flex items-center gap-2">
            <Activity className="h-4 w-4 text-primary" /> System Activity Log
          </h2>
          {activities.length === 0 ? (
            <p className="text-sm text-muted-foreground py-8 text-center">No activity recorded yet.</p>
          ) : (
            <div className="divide-y divide-border">
              {activities.map((act) => (
                <div key={act.id} className="py-3 flex items-start justify-between gap-4 text-xs">
                  <div>
                    <span className="font-semibold text-foreground">{act.eventType}</span>:{" "}
                    <span className="text-muted-foreground">{act.description}</span>
                    {act.project && (
                      <span className="ml-2 rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                        {act.project.name}
                      </span>
                    )}
                  </div>
                  <time className="text-[11px] text-muted-foreground shrink-0">
                    {new Date(act.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                  </time>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* Tasks List Column */}
          <div className="lg:col-span-5 flex flex-col gap-4">
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Filter tasks..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="w-full rounded-lg border border-border bg-background py-1.5 pl-8 pr-3 text-xs placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </div>
              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value)}
                className="rounded-lg border border-border bg-background py-1.5 px-2 text-xs text-foreground focus:outline-none"
              >
                <option value="ALL">All Types</option>
                <option value="RESEARCH">Research</option>
                <option value="ANALYSIS">Analysis</option>
                <option value="DOCUMENT">Document</option>
                <option value="CODING">Coding</option>
                <option value="PROJECT">Project</option>
                <option value="GENERAL">General</option>
              </select>
            </div>

            {loading ? (
              <div className="flex justify-center py-12">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : filteredTasks.length === 0 ? (
              <div className="rounded-xl border border-dashed border-border p-8 text-center text-xs text-muted-foreground">
                No tasks found. Create one to get started!
              </div>
            ) : (
              <div className="space-y-2">
                {filteredTasks.map((t) => (
                  <div
                    key={t.id}
                    onClick={() => {
                      setSelectedTask(t);
                      void loadTaskDetails(t.id);
                    }}
                    className={cn(
                      "cursor-pointer rounded-xl border p-3.5 transition-all text-left",
                      selectedTask?.id === t.id
                        ? "border-primary bg-primary/5 shadow-sm"
                        : "border-border bg-card hover:border-border/80 hover:bg-muted/30",
                    )}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <h3 className="font-semibold text-sm text-foreground line-clamp-1">{t.title}</h3>
                      <span
                        className={cn(
                          "rounded-full px-2 py-0.5 text-[10px] font-semibold shrink-0 uppercase",
                          t.status === "COMPLETED" && "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
                          t.status === "RUNNING" && "bg-blue-500/10 text-blue-600 dark:text-blue-400 animate-pulse",
                          t.status === "FAILED" && "bg-rose-500/10 text-rose-600 dark:text-rose-400",
                          t.status === "PENDING" && "bg-muted text-muted-foreground",
                        )}
                      >
                        {t.status}
                      </span>
                    </div>

                    {t.description && (
                      <p className="mt-1 text-xs text-muted-foreground line-clamp-2">{t.description}</p>
                    )}

                    <div className="mt-2.5 flex items-center gap-2 text-[11px] text-muted-foreground">
                      <span className="rounded bg-muted px-1.5 py-0.5 text-[10px]">{t.type}</span>
                      {t.project && (
                        <span className="rounded bg-primary/10 text-primary px-1.5 py-0.5 text-[10px]">
                          {t.project.name}
                        </span>
                      )}
                      <span className="ml-auto text-[10px]">
                        {new Date(t.updatedAt).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Task Inspector Column */}
          <div className="lg:col-span-7">
            {selectedTask ? (
              <div className="rounded-xl border border-border bg-card p-6 shadow-sm space-y-6">
                <div className="flex items-start justify-between gap-4 border-b border-border pb-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="rounded bg-muted px-2 py-0.5 text-xs font-medium text-foreground">
                        {selectedTask.type}
                      </span>
                      {selectedTask.project && (
                        <span className="rounded bg-primary/10 text-primary px-2 py-0.5 text-xs font-medium">
                          {selectedTask.project.name}
                        </span>
                      )}
                    </div>
                    <h2 className="text-lg font-bold text-foreground mt-2">{selectedTask.title}</h2>
                    {selectedTask.description && (
                      <p className="text-xs text-muted-foreground mt-1">{selectedTask.description}</p>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      disabled={executing || selectedTask.status === "RUNNING"}
                      onClick={() => handleExecuteTask(selectedTask.id)}
                      className="gap-1.5"
                    >
                      {executing ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Play className="h-3.5 w-3.5" />
                      )}
                      {selectedTask.status === "COMPLETED" ? "Re-run Task" : "Run Task"}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleDeleteTask(selectedTask.id)}
                      className="text-destructive hover:bg-destructive/10"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>

                {/* Execution Steps */}
                <div>
                  <h3 className="text-sm font-semibold text-foreground mb-3 flex items-center gap-2">
                    <ListTodo className="h-4 w-4 text-primary" /> Execution Steps
                  </h3>
                  {!selectedTask.steps || selectedTask.steps.length === 0 ? (
                    <p className="text-xs text-muted-foreground italic py-3">
                      No execution steps recorded yet. Click "Run Task" to begin execution.
                    </p>
                  ) : (
                    <div className="space-y-3">
                      {selectedTask.steps.map((step) => (
                        <div
                          key={step.id}
                          className="rounded-lg border border-border/80 bg-muted/20 p-3.5 text-xs space-y-1.5"
                        >
                          <div className="flex items-center justify-between">
                            <span className="font-semibold text-foreground flex items-center gap-1.5">
                              {step.status === "COMPLETED" && (
                                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                              )}
                              {step.status === "RUNNING" && (
                                <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />
                              )}
                              {step.status === "FAILED" && (
                                <XCircle className="h-3.5 w-3.5 text-rose-500" />
                              )}
                              {step.status === "PENDING" && (
                                <Clock className="h-3.5 w-3.5 text-muted-foreground" />
                              )}
                              Step {step.stepNumber}: {step.title}
                            </span>
                            {step.toolName && (
                              <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground">
                                tool: {step.toolName}
                              </span>
                            )}
                          </div>
                          {step.description && (
                            <p className="text-muted-foreground">{step.description}</p>
                          )}
                          {step.error && (
                            <p className="text-rose-600 dark:text-rose-400 font-medium">
                              Error: {step.error}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Execution Result Summary */}
                {selectedTask.resultSummary && (
                  <div className="rounded-lg border border-border bg-background p-4 space-y-2">
                    <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider">
                      Execution Result &amp; Summary
                    </h3>
                    <div className="text-xs text-foreground/90 whitespace-pre-wrap font-mono leading-relaxed bg-muted/30 p-3 rounded border border-border/60">
                      {selectedTask.resultSummary}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-border text-xs text-muted-foreground">
                Select a task to inspect execution details or run tasks.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Create Task Modal */}
      {isCreating && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="w-full max-w-lg rounded-2xl border border-border bg-card p-6 shadow-xl space-y-4">
            <h2 className="text-lg font-bold text-foreground">Create Autonomous Task</h2>
            <form onSubmit={handleCreateTask} className="space-y-4">
              <div>
                <label className="text-xs font-medium text-foreground">Task Title *</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Research Next.js 15 features and evaluate performance"
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </div>

              <div>
                <label className="text-xs font-medium text-foreground">Description &amp; Goals</label>
                <textarea
                  rows={3}
                  placeholder="Provide additional context, requirements, or focus areas..."
                  value={newDesc}
                  onChange={(e) => setNewDesc(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-xs focus:outline-none focus:ring-1 focus:ring-primary"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-medium text-foreground">Task Type</label>
                  <select
                    value={newType}
                    onChange={(e) => setNewType(e.target.value)}
                    className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-xs focus:outline-none"
                  >
                    <option value="GENERAL">General</option>
                    <option value="RESEARCH">Web Research</option>
                    <option value="ANALYSIS">Data Analysis</option>
                    <option value="DOCUMENT">Document Synthesis</option>
                    <option value="CODING">Coding &amp; Scripting</option>
                    <option value="PROJECT">Project Scoped</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs font-medium text-foreground">Priority</label>
                  <select
                    value={newPriority}
                    onChange={(e) => setNewPriority(e.target.value)}
                    className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-xs focus:outline-none"
                  >
                    <option value="LOW">Low</option>
                    <option value="NORMAL">Normal</option>
                    <option value="HIGH">High</option>
                    <option value="URGENT">Urgent</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="text-xs font-medium text-foreground">Project Workspace (Optional)</label>
                <select
                  value={newProjectId}
                  onChange={(e) => setNewProjectId(e.target.value)}
                  className="mt-1 w-full rounded-lg border border-border bg-background px-3 py-2 text-xs focus:outline-none"
                >
                  <option value="">No Project (Global Workspace)</option>
                  {projects.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <Button type="button" variant="ghost" size="sm" onClick={() => setIsCreating(false)}>
                  Cancel
                </Button>
                <Button type="submit" size="sm" disabled={savingTask || !newTitle.trim()}>
                  {savingTask ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Create Task"}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
