"""Extract measurable execution metrics from smolagents memory."""

from dataclasses import dataclass
from typing import Optional

from smolagents.memory import ActionStep


@dataclass
class AgentMetrics:
    agent_step_count: Optional[int] = None
    json_error_count: Optional[int] = None
    tool_call_error_count: Optional[int] = None


def extract_agent_metrics(agent):
    """Inspect completed ActionStep records without altering the agent."""

    memory = getattr(agent, "memory", None)
    steps = getattr(memory, "steps", None)

    if steps is None:
        return AgentMetrics()

    action_steps = [
        step for step in steps
        if isinstance(step, ActionStep)
    ]

    json_errors = 0
    tool_errors = 0

    for step in action_steps:
        if step.error is None:
            continue

        error_text = str(step.error).lower()
        error_type = type(step.error).__name__.lower()

        is_parsing_error = (
            "parsing" in error_type
            or "parse" in error_type
            or "parsing tool call" in error_text
            or "tool call needs to have a key" in error_text
            or "jsondecode" in error_type
            or "invalid json" in error_text
        )

        if is_parsing_error:
            json_errors += 1
        else:
            tool_errors += 1

    return AgentMetrics(
        agent_step_count=len(action_steps),
        json_error_count=json_errors,
        tool_call_error_count=tool_errors,
    )
