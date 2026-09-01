"""Tool safety gates and permission verification."""

from __future__ import annotations

import re
from typing import Any
from bravien.tools.schemas import RiskLevel, ToolDefinition


class ToolPermissionGate:
    """Evaluates whether a tool execution is safe to proceed automatically or requires confirmation."""

    # Never allow execution of system or network arbitrary commands
    FORBIDDEN_OPERATIONS = [
        re.compile(r"\b(eval|exec|__import__|subprocess|os\.system|shutil\.rmtree)\b"),
        re.compile(r"\b(rm\s+-rf|format\s+[a-z]:|drop\s+database)\b", re.IGNORECASE),
        re.compile(r"\b(curl|wget|nc\s+-e|bash\s+-i)\b", re.IGNORECASE),
    ]

    @classmethod
    def check_permission(
        cls,
        tool: ToolDefinition,
        arguments: dict[str, Any],
        user_confirmed: bool = False,
    ) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        # 1. Check for command injection in string arguments
        for k, v in arguments.items():
            if isinstance(v, str):
                for pat in cls.FORBIDDEN_OPERATIONS:
                    if pat.search(v):
                        return False, f"Blocked unsafe argument pattern in parameter '{k}'"

        # 2. Risk check
        if tool.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) or tool.requires_confirmation:
            if not user_confirmed:
                return False, f"Action '{tool.name}' requires explicit confirmation."

        return True, "Allowed"
