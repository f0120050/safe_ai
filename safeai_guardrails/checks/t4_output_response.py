"""T4 output detection with optional incident response."""

from safeai_guardrails.checks.t4_output_check import (
    T4OutputSafetyCheck,
)


class T4OutputResponseCheck:
    def __init__(
        self,
        adapter,
        log_provider,
        incident_response,
        incident_context_provider,
    ):
        self.detector = T4OutputSafetyCheck(adapter, log_provider)
        self.incident_response = incident_response
        self.incident_context_provider = incident_context_provider
        self.last_incident_result = None

    def check(self, response, provenance=None):
        decision = self.detector.check(
            response,
            provenance=provenance,
        )

        if (
            decision.decision == "DENY"
            and decision.reason == "Unauthorized basket item update detected"
        ):
            try:
                context = self.incident_context_provider()
                self.last_incident_result = (
                    self.incident_response.respond(**context)
                )
            except Exception:
                self.last_incident_result = None

        return decision
