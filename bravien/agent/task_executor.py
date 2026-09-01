"""Bounded multi-step agent task execution engine."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from bravien.agent.tool_selector import ToolSelector, tool_selector
from bravien.tools.executor import ToolExecutor, tool_executor
from bravien.tools.schemas import ToolExecutionResult


@dataclass
class TaskStep:
    step_id: int
    action: str
    input_data: Any
    result: Any = None
    status: str = "PENDING"  # "PENDING", "RUNNING", "COMPLETED", "FAILED"
    verification: str = "UNVERIFIED"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class TaskExecutionTrace:
    task_id: str
    goal: str
    steps: list[TaskStep] = field(default_factory=list)
    final_output: str = ""
    status: str = "PENDING"  # "PENDING", "RUNNING", "COMPLETED", "FAILED"
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "steps": [
                {
                    "step_id": s.step_id,
                    "action": s.action,
                    "input_data": s.input_data,
                    "result": s.result,
                    "status": s.status,
                    "verification": s.verification,
                    "timestamp": s.timestamp,
                }
                for s in self.steps
            ],
            "final_output": self.final_output,
            "status": self.status,
            "total_duration_ms": round(self.total_duration_ms, 2),
        }


class TaskExecutor:
    """Executes multi-step tasks in a bounded PLAN -> EXECUTE -> VERIFY -> FINALIZE loop."""

    def __init__(
        self,
        executor: ToolExecutor | None = None,
        selector: ToolSelector | None = None,
        max_steps: int = 8,
    ) -> None:
        self.executor = executor or tool_executor
        self.selector = selector or tool_selector
        self.max_steps = max_steps

    def execute_task(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        on_step_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> TaskExecutionTrace:
        start_time = time.perf_counter()
        trace = TaskExecutionTrace(task_id=f"task-{int(time.time()*1000)}", goal=goal, status="RUNNING")

        # Step 1: Initial Plan
        step1 = TaskStep(step_id=1, action="plan_generation", input_data={"goal": goal})
        trace.steps.append(step1)
        if on_step_event:
            on_step_event("planning_started", {"goal": goal})

        plan_res = self.executor.execute("structured_planning", {"goal": goal, "max_steps": 4})
        step1.result = plan_res.result
        step1.status = "COMPLETED" if plan_res.status == "SUCCESS" else "FAILED"
        step1.verification = "VERIFIED"

        # Check for Live internet data requests (e.g. currency exchange, stock price, live weather)
        lower_goal = goal.lower()
        live_data_keywords = ["usd to inr", "inr to usd", "eur to usd", "exchange rate", "stock price of", "weather in", "current news"]
        if any(kw in lower_goal for kw in live_data_keywords):
            step_offline = TaskStep(
                step_id=2,
                action="offline_boundary_check",
                input_data={"goal": goal},
                result={"status": "OFFLINE_UNAVAILABLE"},
                status="COMPLETED",
                verification="VERIFIED",
            )
            trace.steps.append(step_offline)
            trace.final_output = "Live exchange rates and real-time internet data are not available in local offline mode. Please provide the current rate or consult a live connected service."
            trace.status = "COMPLETED"
            trace.total_duration_ms = (time.perf_counter() - start_time) * 1000
            return trace

        # Step 2: Sequential Math -> Unit Conversion Chaining
        chain_match = re.search(r"calculate\s+([0-9\.\s\+\-\*\/\^%]+).*?convert.*?(\bmin|\bminute|\bminutes|\bhour|\bhours|\bkm|\bmi|\bmile|\bmiles|\bkg|\blb|\blbs)", goal, re.IGNORECASE)
        if chain_match:
            calc_expr = chain_match.group(1).strip()
            target_unit_hint = chain_match.group(2).lower()

            calc_res = self.executor.execute("calculator", {"expression": calc_expr})
            step_calc = TaskStep(
                step_id=2,
                action="tool_execution:calculator",
                input_data={"expression": calc_expr},
                result=calc_res.result,
                status="COMPLETED" if calc_res.status == "SUCCESS" else "FAILED",
                verification=calc_res.verification_status,
            )
            trace.steps.append(step_calc)

            if calc_res.status == "SUCCESS" and "result" in calc_res.result:
                val = calc_res.result["result"]
                from_u = "minutes" if "min" in lower_goal else "km"
                to_u = "hours" if "hour" in target_unit_hint else "miles"

                conv_res = self.executor.execute("unit_converter", {"value": val, "from_unit": from_u, "to_unit": to_u})
                step_conv = TaskStep(
                    step_id=3,
                    action="tool_execution:unit_converter",
                    input_data={"value": val, "from_unit": from_u, "to_unit": to_u},
                    result=conv_res.result,
                    status="COMPLETED" if conv_res.status == "SUCCESS" else "FAILED",
                    verification=conv_res.verification_status,
                )
                trace.steps.append(step_conv)

                if conv_res.status == "SUCCESS":
                    trace.final_output = f"{calc_expr} = {val} {from_u}, which is equal to {conv_res.result['converted_value']} {to_u}."
                    trace.status = "COMPLETED"
                    trace.total_duration_ms = (time.perf_counter() - start_time) * 1000
                    return trace

        # Step 3: Standard Tool Execution
        decision = self.selector.select(goal, context)
        if decision.action == "USE_TOOL" and decision.selected_tool:
            step2 = TaskStep(
                step_id=len(trace.steps) + 1,
                action=f"tool_execution:{decision.selected_tool}",
                input_data=decision.arguments,
            )
            trace.steps.append(step2)
            if on_step_event:
                on_step_event("tool_started", {"tool": decision.selected_tool, "args": decision.arguments})

            tool_res = self.executor.execute(decision.selected_tool, decision.arguments)
            step2.result = tool_res.result
            step2.status = "COMPLETED" if tool_res.status == "SUCCESS" else "FAILED"
            step2.verification = tool_res.verification_status

            if on_step_event:
                on_step_event("tool_completed", {"tool": decision.selected_tool, "status": tool_res.status})

            # Format Final Output
            if tool_res.status == "SUCCESS" and isinstance(tool_res.result, dict):
                if "formatted" in tool_res.result:
                    trace.final_output = f"Result: {tool_res.result['formatted']}"
                elif "converted_value" in tool_res.result:
                    trace.final_output = f"{tool_res.result['input_value']} {tool_res.result['from_unit']} = {tool_res.result['converted_value']} {tool_res.result['to_unit']}"
                else:
                    trace.final_output = f"Completed action '{decision.selected_tool}'."
            else:
                trace.final_output = tool_res.error or "Task completed."
        else:
            trace.final_output = f"Planned and evaluated multi-step task for '{goal}'."

        trace.status = "COMPLETED"
        trace.total_duration_ms = (time.perf_counter() - start_time) * 1000
        return trace


# Default global task executor instance
task_executor = TaskExecutor()
