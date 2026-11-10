"""Layered configuration.

- Input: early filter that can only reject provably unauthorized requests.
  ALLOW and REVIEW both proceed; input never grants authorization.
- Execution: the authorization boundary; the agent must be a
  SafeAIExecutionAgent whose tools are all covered by its checker.
- Output: blocks only on positive violation evidence (DENY);
  REVIEW is recorded but does not block.
"""

from dataclasses import dataclass
from typing import Any, Optional

from safeai_guardrails.checks.execution_guard import (
    SafeAIExecutionAgent,
    SafeAIExecutionDenied,
)
from safeai_guardrails.mediation import assert_complete_mediation


@dataclass
class LayeredResult:
    status: str
    reason: str
    blocked_by: Optional[str] = None
    input_decision: Optional[str] = None
    output_decision: Optional[str] = None
    response: Any = None
    agent_executed: bool = False


class LayeredOrchestrator:
    def __init__(self, input_checker, output_checker=None):
        self.input_checker = input_checker
        self.output_checker = output_checker

    def run(self, task, agent):
        if not isinstance(agent, SafeAIExecutionAgent):
            raise TypeError("Layered mode requires a SafeAIExecutionAgent")
        assert_complete_mediation(agent, agent.execution_checker)

        input_decision = self.input_checker.check(task)
        input_value = str(input_decision.decision)

        if input_value == "DENY":
            return LayeredResult(
                status="BLOCKED",
                reason=f"Input: {input_decision.reason}",
                blocked_by="input",
                input_decision=input_value,
                agent_executed=False,
            )

        try:
            response = agent.run(task)
        except SafeAIExecutionDenied:
            return LayeredResult(
                status="BLOCKED",
                reason="Execution: pre-tool authorization denied",
                blocked_by="execution",
                input_decision=input_value,
                agent_executed=True,
            )

        output_value = None
        if self.output_checker is not None:
            output_decision = self.output_checker.check(response)
            output_value = str(output_decision.decision)
            if output_value == "DENY":
                return LayeredResult(
                    status="BLOCKED",
                    reason=f"Output: {output_decision.reason}",
                    blocked_by="output",
                    input_decision=input_value,
                    output_decision=output_value,
                    agent_executed=True,
                )

        return LayeredResult(
            status="ALLOWED",
            reason="Layered checks passed",
            input_decision=input_value,
            output_decision=output_value,
            response=response,
            agent_executed=True,
        )
