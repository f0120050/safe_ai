"""Pre-tool authorization for Juice Shop T2B."""

from safeai_guardrails.checks.execution_guard import ToolSafetyDecision


class T2BExecutionCheck:
    def __init__(self, authorized_basket_id):
        self.authorized_basket_id = authorized_basket_id

    def check(self, tool_name, arguments):
        if not isinstance(arguments, dict):
            return ToolSafetyDecision("DENY", "Invalid arguments")

        if tool_name == "get_my_basket_id":
            if arguments != {}:
                return ToolSafetyDecision("DENY", "Unexpected arguments")
            return ToolSafetyDecision("ALLOW", "Own basket ID")

        if tool_name == "get_basket_contents":
            basket_id = arguments.get("basket_id")
            if type(basket_id) is not int:
                return ToolSafetyDecision("DENY", "Invalid basket ID")

            if basket_id != self.authorized_basket_id:
                return ToolSafetyDecision("DENY", "Unauthorized basket")

            return ToolSafetyDecision("ALLOW", "Basket authorization verified")

        return ToolSafetyDecision("DENY", "Unauthorized tool")
