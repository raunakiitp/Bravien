"""Masked and structured tool errors."""

from __future__ import annotations


class ToolError(Exception):
    """Base error for tool execution."""

    def __init__(self, message: str, code: str = "TOOL_ERROR", safe_user_message: str | None = None):
        super().__init__(message)
        self.code = code
        self.safe_user_message = safe_user_message or "Tool execution could not be completed."


class ToolNotFoundError(ToolError):
    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' is not registered.", code="TOOL_NOT_FOUND", safe_user_message=f"The requested capability '{tool_name}' is unavailable.")


class ToolPermissionDeniedError(ToolError):
    def __init__(self, tool_name: str, reason: str = "Confirmation required"):
        super().__init__(f"Permission denied for '{tool_name}': {reason}", code="PERMISSION_DENIED", safe_user_message="Action requires explicit user authorization.")


class ToolTimeoutError(ToolError):
    def __init__(self, tool_name: str, timeout: float):
        super().__init__(f"Tool '{tool_name}' timed out after {timeout}s", code="TOOL_TIMEOUT", safe_user_message="Operation took too long and was safely aborted.")


class ToolValidationError(ToolError):
    def __init__(self, message: str):
        super().__init__(message, code="INVALID_ARGUMENTS", safe_user_message="Invalid parameters provided for operation.")
