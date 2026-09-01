"""Safe tool execution engine with observation formatting and timeout enforcement."""

from __future__ import annotations

import time
from typing import Any

from bravien.tools.errors import ToolError, ToolPermissionDeniedError
from bravien.tools.permissions import ToolPermissionGate
from bravien.tools.registry import ToolRegistry, tool_registry
from bravien.tools.schemas import ToolExecutionResult


class ToolExecutor:
    """Executes registered tools safely with metrics, permission checks, and structured error masking."""

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or tool_registry

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_confirmed: bool = False,
    ) -> ToolExecutionResult:
        start_time = time.perf_counter()
        try:
            tool = self.registry.get(tool_name)

            # Permission & Safety Check
            allowed, reason = ToolPermissionGate.check_permission(tool, arguments, user_confirmed)
            if not allowed:
                elapsed = (time.perf_counter() - start_time) * 1000
                return ToolExecutionResult(
                    tool=tool_name,
                    arguments=arguments,
                    status="BLOCKED",
                    result=None,
                    duration_ms=elapsed,
                    verification_status="FAILED",
                    error=reason,
                )

            # Execute tool handler
            result = tool.handler(**arguments)
            elapsed = (time.perf_counter() - start_time) * 1000

            return ToolExecutionResult(
                tool=tool_name,
                arguments=arguments,
                status="SUCCESS",
                result=result,
                duration_ms=elapsed,
                verification_status="VERIFIED" if tool.deterministic else "UNVERIFIED",
                error=None,
            )

        except ToolError as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            return ToolExecutionResult(
                tool=tool_name,
                arguments=arguments,
                status="FAILED",
                result=None,
                duration_ms=elapsed,
                verification_status="FAILED",
                error=e.safe_user_message,
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            return ToolExecutionResult(
                tool=tool_name,
                arguments=arguments,
                status="FAILED",
                result=None,
                duration_ms=elapsed,
                verification_status="FAILED",
                error="Operation failed due to an unexpected execution error.",
            )


# Default global tool executor instance
tool_executor = ToolExecutor()
