"""Input authorization check for Juice Shop basket reading."""

import re
from safeai_guardrails.checks.input_check import SafetyDecision


BASKET_PATTERN = re.compile(
    r"\bbasket\s*(?:id\s*[:#]?\s*)?(\d+)\b",
    re.IGNORECASE,
)


class JuiceShopReadInputCheck:
    def __init__(self, authorized_basket_id):
        self.authorized_basket_id = int(authorized_basket_id)

    def check(self, task):
        if not isinstance(task, str) or not task.strip():
            return SafetyDecision("DENY", "Invalid task", [])

        basket_ids = list(dict.fromkeys(
            match.group(1)
            for match in BASKET_PATTERN.finditer(task)
        ))

        if not basket_ids:
            return SafetyDecision(
                "REVIEW",
                "No explicit basket ID; authorization not established",
                [],
            )

        unauthorized = [
            basket_id for basket_id in basket_ids
            if int(basket_id) != self.authorized_basket_id
        ]

        if unauthorized:
            return SafetyDecision(
                "DENY",
                "Task references a basket outside the authorized session",
                basket_ids,
            )

        return SafetyDecision(
            "ALLOW",
            "All referenced baskets are authorized",
            basket_ids,
        )
