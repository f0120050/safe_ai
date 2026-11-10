from dataclasses import dataclass


@dataclass
class ExecutionSafetyDecision:
    decision: str
    reason: str
    checked_resource_ids: list[str]


class ExecutionOutputCheck:
    """
    Verify resources accessed by an agent using
    trusted execution records and an authorization API.

    This check runs AFTER agent execution.
    """

    def __init__(self, adapter):
        self.adapter = adapter

    def check(self, response, execution_log):
        if not isinstance(response, str) or not response.strip():
            return ExecutionSafetyDecision(
                "DENY",
                "Empty or invalid agent response",
                []
            )

        if not isinstance(execution_log, list):
            return ExecutionSafetyDecision(
                "DENY",
                "Missing or invalid execution log",
                []
            )

        accessed_ids = set()

        for event in execution_log:
            if not isinstance(event, dict):
                return ExecutionSafetyDecision(
                    "DENY",
                    "Malformed execution event",
                    []
                )

            if event.get("tool") == "get_vehicle_location":
                uuid = event.get("requested_uuid")

                if not isinstance(uuid, str) or not uuid.strip():
                    return ExecutionSafetyDecision(
                        "DENY",
                        "Missing vehicle UUID in execution log",
                        []
                    )

                accessed_ids.add(uuid.lower())

        if not accessed_ids:
            return ExecutionSafetyDecision(
                "REVIEW",
                "No vehicle location access found in execution log",
                []
            )

        try:
            authorized_ids = {
                uuid.lower()
                for uuid in self.adapter.get_authorized_vehicle_uuids()
            }
        except Exception:
            return ExecutionSafetyDecision(
                "DENY",
                "Authorization API verification failed",
                sorted(accessed_ids)
            )

        unauthorized_ids = accessed_ids - authorized_ids

        if unauthorized_ids:
            return ExecutionSafetyDecision(
                "DENY",
                "Agent accessed resources not authorized for the user",
                sorted(accessed_ids)
            )

        return ExecutionSafetyDecision(
            "ALLOW",
            "All recorded vehicle accesses are authorized",
            sorted(accessed_ids)
        )
