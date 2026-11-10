from dataclasses import dataclass

from safeai_guardrails.checks.input_check import UUID_PATTERN


@dataclass
class OutputSafetyDecision:
    decision: str
    reason: str


class OutputSafetyCheck:
    def __init__(self, adapter):
        self.adapter = adapter

    def check(self, response, provenance=None):
        if not isinstance(response, str) or not response.strip():
            return OutputSafetyDecision(
                "DENY", "Empty or invalid output"
            )

        resource_ids = {
            uuid.lower()
            for uuid in UUID_PATTERN.findall(response)
        }

        # Explicit resource IDs can be checked independently.
        if resource_ids:
            try:
                authorized = {
                    uuid.lower()
                    for uuid in self.adapter.get_authorized_vehicle_uuids()
                }
            except Exception:
                return OutputSafetyDecision(
                    "DENY", "Output authorization verification failed"
                )

            if not resource_ids.issubset(authorized):
                return OutputSafetyDecision(
                    "DENY", "Output references unauthorized resources"
                )

        # Without trusted provenance, absence of UUIDs does not
        # establish that the response contains only authorized data.
        if provenance is None:
            return OutputSafetyDecision(
                "REVIEW", "Output provenance is not verified"
            )

        if not isinstance(provenance, dict):
            return OutputSafetyDecision(
                "DENY", "Invalid provenance"
            )

        if provenance.get("verified") is not True:
            return OutputSafetyDecision(
                "DENY", "Output provenance is not verified"
            )

        source_ids = provenance.get("resource_ids")

        if not isinstance(source_ids, list):
            return OutputSafetyDecision(
                "DENY", "Invalid provenance resource list"
            )

        if not all(isinstance(x, str) for x in source_ids):
            return OutputSafetyDecision(
                "DENY", "Invalid provenance resource identifier"
            )

        try:
            authorized = {
                uuid.lower()
                for uuid in self.adapter.get_authorized_vehicle_uuids()
            }
        except Exception:
            return OutputSafetyDecision(
                "DENY", "Output authorization verification failed"
            )

        if not {x.lower() for x in source_ids}.issubset(authorized):
            return OutputSafetyDecision(
                "DENY", "Output contains unauthorized resource provenance"
            )

        return OutputSafetyDecision(
            "ALLOW", "Output resources are authorized"
        )
