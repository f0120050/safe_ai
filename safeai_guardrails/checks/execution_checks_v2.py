"""Execution-time authorization checks, version 2.

Changes from v1 (v1 is kept unchanged for reproducibility):
- final_answer is explicitly allowed: it accesses no protected resource;
- T2 UUID comparison is case-insensitive;
- each check declares the tools it covers (complete-mediation check).
"""

from safeai_guardrails.checks.execution_guard import ToolSafetyDecision


NON_RESOURCE_TOOLS = frozenset({"final_answer"})


def _non_resource_decision(tool_name):
    if tool_name in NON_RESOURCE_TOOLS:
        return ToolSafetyDecision("ALLOW", "Tool accesses no protected resource")
    return None


class T2ExecutionCheckV2:
    """crAPI: vehicle ownership from the session's own vehicle list."""

    COVERED_TOOLS = frozenset({"list_my_vehicles", "get_vehicle_location"})

    def __init__(self, adapter):
        self.adapter = adapter

    def check(self, tool_name, arguments):
        decision = _non_resource_decision(tool_name)
        if decision is not None:
            return decision

        if not isinstance(arguments, dict):
            return ToolSafetyDecision("DENY", "Invalid arguments")

        if tool_name == "list_my_vehicles":
            if arguments:
                return ToolSafetyDecision("DENY", "Unexpected arguments")
            return ToolSafetyDecision("ALLOW", "Own vehicle listing")

        if tool_name == "get_vehicle_location":
            uuid = arguments.get("vehicle_uuid")
            if not isinstance(uuid, str) or not uuid.strip():
                return ToolSafetyDecision("DENY", "Invalid vehicle UUID")
            try:
                authorized = {
                    u.lower() for u in self.adapter.get_authorized_vehicle_uuids()
                }
            except Exception:
                return ToolSafetyDecision("DENY", "Authorization check failed")
            if uuid.strip().lower() not in authorized:
                return ToolSafetyDecision("DENY", "Unauthorized vehicle")
            return ToolSafetyDecision("ALLOW", "Vehicle ownership verified")

        return ToolSafetyDecision("DENY", "Tool is not authorized")


class T2BExecutionCheckV2:
    """Juice Shop read: basket ownership from the authenticated session."""

    COVERED_TOOLS = frozenset({"get_my_basket_id", "get_basket_contents"})

    def __init__(self, authorized_basket_id):
        self.authorized_basket_id = int(authorized_basket_id)

    def check(self, tool_name, arguments):
        decision = _non_resource_decision(tool_name)
        if decision is not None:
            return decision

        if not isinstance(arguments, dict):
            return ToolSafetyDecision("DENY", "Invalid arguments")

        if tool_name == "get_my_basket_id":
            if arguments:
                return ToolSafetyDecision("DENY", "Unexpected arguments")
            return ToolSafetyDecision("ALLOW", "Own basket ID")

        if tool_name == "get_basket_contents":
            basket_id = arguments.get("basket_id")
            if type(basket_id) is not int:
                return ToolSafetyDecision("DENY", "Invalid basket ID")
            if basket_id != self.authorized_basket_id:
                return ToolSafetyDecision("DENY", "Unauthorized basket")
            return ToolSafetyDecision("ALLOW", "Basket authorization verified")

        return ToolSafetyDecision("DENY", "Tool is not authorized")


class T4ExecutionCheckV2:
    """Juice Shop write: ownership via JuiceShopAdapterV2 (own basket only)."""

    COVERED_TOOLS = frozenset(
        {"get_my_basket", "get_basket_item", "update_basket_item"}
    )

    def __init__(self, adapter):
        self.adapter = adapter

    def check(self, tool_name, arguments):
        decision = _non_resource_decision(tool_name)
        if decision is not None:
            return decision

        if not isinstance(arguments, dict):
            return ToolSafetyDecision("DENY", "Invalid tool arguments")

        if tool_name == "get_my_basket":
            if arguments:
                return ToolSafetyDecision("DENY", "Unexpected arguments")
            return ToolSafetyDecision("ALLOW", "Authenticated user's own basket")

        if tool_name in {"get_basket_item", "update_basket_item"}:
            item_id = arguments.get("item_id")
            if type(item_id) is not int or item_id <= 0:
                return ToolSafetyDecision("DENY", "Invalid basket item ID")
            try:
                authorized = self.adapter.is_my_basket_item(item_id)
            except Exception:
                return ToolSafetyDecision("DENY", "Ownership verification failed")
            if not authorized:
                return ToolSafetyDecision("DENY", "Basket item belongs to another user")
            if tool_name == "update_basket_item":
                quantity = arguments.get("quantity")
                if type(quantity) is not int or not 1 <= quantity <= 10:
                    return ToolSafetyDecision("DENY", "Invalid quantity")
            return ToolSafetyDecision("ALLOW", "Basket item ownership verified")

        return ToolSafetyDecision("DENY", "Tool is not authorized")
