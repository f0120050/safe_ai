"""Independent pre-execution authorization for T4."""

import re

from safeai_guardrails.checks.input_check import SafetyDecision


ITEM_PATTERN = re.compile(
    r"\bbasket\s*item\s*(?:id\s*[:#]?\s*)?(\d+)\b",
    re.IGNORECASE,
)


class T4InputSafetyCheck:
    def __init__(self, adapter):
        self.adapter = adapter

    def check(self, task):
        if not isinstance(task, str) or not task.strip():
            return SafetyDecision("DENY", "Invalid task", [])

        item_ids = list(dict.fromkeys(
            match.group(1)
            for match in ITEM_PATTERN.finditer(task)
        ))

        if not item_ids:
            return SafetyDecision(
                "REVIEW",
                "No explicit basket item ID; authorization not established",
                [],
            )

        try:
            unauthorized = [
                item_id
                for item_id in item_ids
                if not self.adapter.is_my_basket_item(int(item_id))
            ]
        except Exception:
            return SafetyDecision(
                "DENY",
                "Basket item authorization verification failed",
                item_ids,
            )

        if unauthorized:
            return SafetyDecision(
                "DENY",
                "One or more basket items are not authorized",
                item_ids,
            )

        return SafetyDecision(
            "ALLOW",
            "All specified basket items are authorized",
            item_ids,
        )
