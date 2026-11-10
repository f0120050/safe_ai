"""Execution-time authorization for Juice Shop T4."""

from safeai_guardrails.checks.execution_guard import ToolSafetyDecision


class T4ExecutionCheck:
    def __init__(self, adapter):
        self.adapter = adapter

    def check(self, tool_name, arguments):
        if tool_name == "get_my_basket":
            if arguments != {}:
                return ToolSafetyDecision(
                    "DENY", "Unexpected arguments"
                )
            return ToolSafetyDecision(
                "ALLOW", "Authenticated user's own basket"
            )

        if tool_name in {"get_basket_item", "update_basket_item"}:
            if not isinstance(arguments, dict):
                return ToolSafetyDecision(
                    "DENY", "Invalid tool arguments"
                )

            item_id = arguments.get("item_id")
            if type(item_id) is not int or item_id <= 0:
                return ToolSafetyDecision(
                    "DENY", "Invalid basket item ID"
                )

            try:
                authorized = self.adapter.is_my_basket_item(item_id)
            except Exception:
                return ToolSafetyDecision(
                    "DENY", "Ownership verification failed"
                )

            if not authorized:
                return ToolSafetyDecision(
                    "DENY", "Basket item belongs to another user"
                )

            if tool_name == "update_basket_item":
                quantity = arguments.get("quantity")
                if type(quantity) is not int or not 1 <= quantity <= 10:
                    return ToolSafetyDecision(
                        "DENY", "Invalid quantity"
                    )

            return ToolSafetyDecision(
                "ALLOW", "Basket item ownership verified"
            )

        return ToolSafetyDecision(
            "DENY", "Tool is not authorized"
        )

    def after(self, tool_name, arguments, result):
        return ToolSafetyDecision(
            "ALLOW", "Post-tool checkpoint completed"
        )
