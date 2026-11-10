"""Independent safety modes for Juice Shop unauthorized basket reading."""

from safeai_guardrails.orchestrator import SafeAIOrchestrator
from safeai_guardrails.checks.juice_shop_read_input_check import (
    JuiceShopReadInputCheck,
)
from safeai_guardrails.checks.juice_shop_read_output_check import (
    JuiceShopReadOutputCheck,
)
from safeai_experiments.runners.safety_modes import AllowAllInput


def build_juice_shop_read_orchestrator(
    mode,
    authorized_basket_id,
    log_provider,
):
    supported = {
        "prompt_only",
        "input_only",
        "output_only",
        "input_output",
    }

    if mode not in supported:
        raise ValueError(f"Unsupported safety mode: {mode}")

    input_checker = (
        JuiceShopReadInputCheck(authorized_basket_id)
        if mode in {"input_only", "input_output"}
        else AllowAllInput()
    )

    output_checker = (
        JuiceShopReadOutputCheck(authorized_basket_id, log_provider)
        if mode in {"output_only", "input_output"}
        else None
    )

    return SafeAIOrchestrator(
        input_checker=input_checker,
        output_checker=output_checker,
    )
