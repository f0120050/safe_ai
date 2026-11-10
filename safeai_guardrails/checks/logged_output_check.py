from safeai_guardrails.checks.execution_output_check import ExecutionOutputCheck
from safeai_guardrails.checks.input_check import UUID_PATTERN


class LoggedOutputCheck:
    """
    Output check using application-supplied execution logs.

    The log_provider must come from trusted application code,
    never from LLM-generated text.
    """

    def __init__(self, adapter, log_provider):
        self.adapter = adapter
        self.log_provider = log_provider
        self.execution_checker = ExecutionOutputCheck(adapter)

    def check(self, response, provenance=None):
        try:
            execution_log = self.log_provider()
        except Exception:
            return self.execution_checker.check(response, None)

        result = self.execution_checker.check(
            response,
            execution_log
        )

        if result.decision != "ALLOW":
            return result

        # Also check explicit UUIDs in the final response.
        output_ids = {
            uuid.lower()
            for uuid in UUID_PATTERN.findall(response)
        }

        if output_ids:
            try:
                authorized = {
                    uuid.lower()
                    for uuid in self.adapter.get_authorized_vehicle_uuids()
                }
            except Exception:
                return self.execution_checker.check(response, None)

            if not output_ids.issubset(authorized):
                result.decision = "DENY"
                result.reason = "Output references unauthorized resources"

        return result
