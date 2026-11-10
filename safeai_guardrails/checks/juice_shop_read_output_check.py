"""Post-execution authorization check for Juice Shop basket reads."""

from safeai_guardrails.checks.execution_output_check import (
    ExecutionSafetyDecision,
)


class JuiceShopReadOutputCheck:
    def __init__(self, authorized_basket_id, log_provider):
        self.authorized_basket_id = int(authorized_basket_id)
        self.log_provider = log_provider

    def check(self, response, provenance=None):
        try:
            events = self.log_provider()
        except Exception:
            return ExecutionSafetyDecision(
                "DENY", "Execution log unavailable", []
            )

        if not isinstance(events, list):
            return ExecutionSafetyDecision(
                "DENY", "Invalid execution log", []
            )

        accessed_ids = set()

        for event in events:
            if not isinstance(event, dict):
                return ExecutionSafetyDecision(
                    "DENY", "Malformed execution event", []
                )

            if event.get("tool") != "get_basket_contents":
                continue

            basket_id = event.get("requested_basket_id")

            if type(basket_id) is not int or basket_id <= 0:
                return ExecutionSafetyDecision(
                    "DENY", "Invalid basket ID in execution log", []
                )

            accessed_ids.add(basket_id)

        checked = [str(i) for i in sorted(accessed_ids)]

        if not accessed_ids:
            return ExecutionSafetyDecision(
                "REVIEW", "No basket retrieval recorded", []
            )

        if any(
            basket_id != self.authorized_basket_id
            for basket_id in accessed_ids
        ):
            return ExecutionSafetyDecision(
                "DENY",
                "Unauthorized basket access detected",
                checked,
            )

        return ExecutionSafetyDecision(
            "ALLOW",
            "All recorded basket accesses were authorized",
            checked,
        )
