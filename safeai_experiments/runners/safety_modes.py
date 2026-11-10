"""Configure independent SafeAI security modes."""

from types import SimpleNamespace

from safeai_guardrails.adapters.crapi_adapter import CrapiAdapter
from safeai_guardrails.checks.input_check import InputSafetyCheck
from safeai_guardrails.checks.logged_output_check import LoggedOutputCheck
from safeai_guardrails.orchestrator import SafeAIOrchestrator


class AllowAllInput:
    """No input guardrail: controlled experimental ablation."""

    def check(self, task):
        return SimpleNamespace(
            decision="ALLOW",
            reason="Input safety check disabled for experiment",
        )


def build_orchestrator(mode, session, base_url, log_provider):
    """Return an orchestrator with the requested safety layers."""

    supported = {
        "prompt_only",
        "input_only",
        "output_only",
        "input_output",
    }

    if mode not in supported:
        raise ValueError(f"Unsupported safety mode: {mode}")

    adapter = CrapiAdapter(base_url, session)

    input_checker = (
        InputSafetyCheck(adapter)
        if mode in {"input_only", "input_output"}
        else AllowAllInput()
    )

    output_checker = (
        LoggedOutputCheck(adapter, log_provider)
        if mode in {"output_only", "input_output"}
        else None
    )

    return SafeAIOrchestrator(
        input_checker=input_checker,
        output_checker=output_checker,
    )
