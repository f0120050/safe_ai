"""Independent execution-time authorization for agent tool calls."""

from dataclasses import dataclass
from smolagents import ToolCallingAgent


class SafeAIExecutionDenied(BaseException):
    """Terminal security denial; never retry as an agent tool error."""



@dataclass
class ToolSafetyDecision:
    decision: str
    reason: str


class SafeAIExecutionAgent(ToolCallingAgent):
    def __init__(self, *args, execution_checker, **kwargs):
        super().__init__(*args, **kwargs)
        self.execution_checker = execution_checker
        self.execution_audit = []
        self.execution_blocked = False

    def execute_tool_call(self, tool_name, arguments):
        # Resolve agent state references before authorization.
        resolved = self._substitute_state_variables(arguments)

        decision = self.execution_checker.check(tool_name, resolved)

        if decision.decision not in ("ALLOW", "DENY"):
            decision = ToolSafetyDecision(
                "DENY", "Execution authorization not established"
            )

        self.execution_audit.append({
            "tool": tool_name,
            "decision": decision.decision,
            "reason": decision.reason,
        })

        if decision.decision != "ALLOW":
            self.execution_blocked = True
            raise SafeAIExecutionDenied(
                "SafeAI Execution Security blocked the tool call"
            )

        # Pass resolved arguments to the original tool executor.
        result = super().execute_tool_call(tool_name, resolved)

        # The post-tool hook can inspect execution outcomes.
        post_check = getattr(self.execution_checker, "after", None)

        if callable(post_check):
            post_decision = post_check(tool_name, resolved, result)

            self.execution_audit.append({
                "tool": tool_name,
                "phase": "post",
                "decision": post_decision.decision,
                "reason": post_decision.reason,
            })

            if post_decision.decision != "ALLOW":
                self.execution_blocked = True
                raise SafeAIExecutionDenied(
                    "SafeAI post-tool execution check blocked continuation"
                )

        return result
