"""Startup check: every tool the agent can call must have a policy rule."""

from safeai_guardrails.checks.execution_checks_v2 import NON_RESOURCE_TOOLS


def assert_complete_mediation(agent, execution_checker):
    covered = set(getattr(execution_checker, "COVERED_TOOLS", ()))
    if not covered:
        raise RuntimeError("Execution checker declares no covered tools")

    agent_tools = set(getattr(agent, "tools", {}) or {})
    missing = agent_tools - NON_RESOURCE_TOOLS - covered

    if missing:
        raise RuntimeError(
            f"Tools without an authorization rule: {sorted(missing)}"
        )
