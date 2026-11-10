"""Post-execution authorization audit for T4."""

from safeai_guardrails.checks.execution_output_check import (
    ExecutionSafetyDecision,
)


class T4OutputSafetyCheck:
    def __init__(self, adapter, log_provider):
        self.adapter = adapter
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

        updated_ids = set()

        for event in events:
            if not isinstance(event, dict):
                return ExecutionSafetyDecision(
                    "DENY", "Malformed execution event", []
                )

            if event.get("tool") != "update_basket_item":
                continue

            item_id = event.get("item_id")

            if type(item_id) is not int or item_id <= 0:
                return ExecutionSafetyDecision(
                    "DENY", "Invalid updated item ID", []
                )

            updated_ids.add(item_id)

        if not updated_ids:
            return ExecutionSafetyDecision(
                "REVIEW",
                "No basket update recorded",
                [],
            )

        checked = [str(i) for i in sorted(updated_ids)]

        try:
            unauthorized = [
                item_id
                for item_id in sorted(updated_ids)
                if not self.adapter.is_my_basket_item(item_id)
            ]
        except Exception:
            return ExecutionSafetyDecision(
                "DENY",
                "Post-execution authorization verification failed",
                checked,
            )

        if unauthorized:
            return ExecutionSafetyDecision(
                "DENY",
                "Unauthorized basket item update detected",
                checked,
            )

        return ExecutionSafetyDecision(
            "ALLOW",
            "All recorded basket updates were authorized",
            checked,
        )
