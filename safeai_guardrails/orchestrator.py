from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class AgentExecutionResult:
    response: str
    provenance: dict | None = None


@dataclass
class GuardrailResult:
    status: str
    reason: str
    response: Any = None
    agent_executed: bool = False


class SafeAIOrchestrator:
    def __init__(self, input_checker, output_checker=None):
        self.input_checker = input_checker
        self.output_checker = output_checker

    def run(self, task: str, agent_runner: Callable[[str], Any]):
        input_decision = self.input_checker.check(task)

        if input_decision.decision != "ALLOW":
            return GuardrailResult(
                status="BLOCKED",
                reason=f"Input safety: {input_decision.reason}",
                agent_executed=False
            )

        agent_result = agent_runner(task)

        # An explicit, application-supplied result may contain
        # a response and trusted provenance metadata.
        if isinstance(agent_result, AgentExecutionResult):
            agent_response = agent_result.response
            provenance = agent_result.provenance
        else:
            agent_response = agent_result
            provenance = None

        if self.output_checker is not None:
            output_decision = self.output_checker.check(
                agent_response,
                provenance=provenance
            )

            if output_decision.decision != "ALLOW":
                return GuardrailResult(
                    status="BLOCKED",
                    reason=f"Output safety: {output_decision.reason}",
                    agent_executed=True
                )

        return GuardrailResult(
            status="ALLOWED",
            reason="Safety checks passed",
            response=agent_response,
            agent_executed=True
        )
