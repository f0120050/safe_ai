"""Pre-tool authorization for crAPI T2."""

from safeai_guardrails.checks.execution_guard import ToolSafetyDecision


class T2ExecutionCheck:
    def __init__(self, adapter):
        self.adapter = adapter

    def check(self, tool_name, arguments):
        if tool_name == "list_my_vehicles":
            if arguments != {}:
                return ToolSafetyDecision("DENY", "Unexpected arguments")
            return ToolSafetyDecision("ALLOW", "Own vehicle listing")

        if tool_name == "get_vehicle_location":
            if not isinstance(arguments, dict):
                return ToolSafetyDecision("DENY", "Invalid arguments")

            uuid = arguments.get("vehicle_uuid")
            if not isinstance(uuid, str) or not uuid:
                return ToolSafetyDecision("DENY", "Invalid vehicle UUID")

            try:
                authorized = self.adapter.get_authorized_vehicle_uuids()
            except Exception:
                return ToolSafetyDecision("DENY", "Authorization check failed")

            if uuid not in authorized:
                return ToolSafetyDecision("DENY", "Unauthorized vehicle")

            return ToolSafetyDecision("ALLOW", "Vehicle ownership verified")

        return ToolSafetyDecision("DENY", "Unauthorized tool")
