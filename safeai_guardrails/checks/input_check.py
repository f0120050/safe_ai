import re
from dataclasses import dataclass


UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}\b"
)


@dataclass
class SafetyDecision:
    decision: str
    reason: str
    resource_ids: list[str]


class InputSafetyCheck:
    def __init__(self, adapter):
        self.adapter = adapter

    def check(self, task: str) -> SafetyDecision:
        if not isinstance(task, str) or not task.strip():
            return SafetyDecision(
                "DENY", "Invalid task", []
            )

        resource_ids = list(dict.fromkeys(
            match.lower()
            for match in UUID_PATTERN.findall(task)
        ))

        if not resource_ids:
            return SafetyDecision(
                "REVIEW",
                "No explicit UUID found; authorization not established",
                []
            )

        try:
            authorized = {
                uuid.lower()
                for uuid in self.adapter.get_authorized_vehicle_uuids()
            }
        except Exception:
            return SafetyDecision(
                "DENY",
                "Authorization verification failed",
                resource_ids
            )

        unauthorized = [
            uuid for uuid in resource_ids
            if uuid not in authorized
        ]

        if unauthorized:
            return SafetyDecision(
                "DENY",
                "One or more resource UUIDs are not authorized",
                resource_ids
            )

        return SafetyDecision(
            "ALLOW",
            "All detected resource UUIDs are authorized",
            resource_ids
        )
