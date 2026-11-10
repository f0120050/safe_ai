import unittest

from safeai_guardrails.adapters.juice_shop_adapter_v2 import JuiceShopAdapterV2
from safeai_guardrails.checks.execution_checks_v2 import (
    T2ExecutionCheckV2, T2BExecutionCheckV2, T4ExecutionCheckV2,
)
from safeai_guardrails.checks.execution_guard import (
    SafeAIExecutionAgent, SafeAIExecutionDenied,
)
from safeai_guardrails.checks.input_check import SafetyDecision
from safeai_guardrails.layered_orchestrator import LayeredOrchestrator
from safeai_guardrails.mediation import assert_complete_mediation


OWN = "08ca470e-dee0-4419-8671-2cc7e16bc44d"
OTHER = "7919479f-8f90-4f07-a786-07721111eafd"


class FakeCrapiAdapter:
    def get_authorized_vehicle_uuids(self):
        return {OWN}


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": self._data}


class FakeSession:
    """Serves only the authorized basket; records every URL requested."""

    def __init__(self, basket):
        self.basket = basket
        self.urls = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        if url.endswith(f"/rest/basket/{self.basket['id']}"):
            return FakeResponse(self.basket)
        raise AssertionError(f"Unexpected request: {url}")


def own_basket(items):
    return {
        "id": 7,
        "Products": [
            {"id": pid, "BasketItem": {"id": iid, "BasketId": 7, "quantity": 1}}
            for pid, iid in items
        ],
    }


def fake_guarded_agent(checker, tool_names, behavior):
    agent = SafeAIExecutionAgent.__new__(SafeAIExecutionAgent)
    agent.execution_checker = checker
    agent.execution_audit = []
    agent.tools = {name: None for name in tool_names}
    agent.run = behavior
    return agent


class Decision:
    def __init__(self, decision, reason="test"):
        self.decision = decision
        self.reason = reason


class FixedChecker:
    def __init__(self, decision):
        self.decision = decision

    def check(self, *args, **kwargs):
        return Decision(self.decision)


class TestExecutionChecksV2(unittest.TestCase):
    def test_final_answer_allowed_everywhere(self):
        for checker in (
            T2ExecutionCheckV2(FakeCrapiAdapter()),
            T2BExecutionCheckV2(7),
            T4ExecutionCheckV2(None),
        ):
            self.assertEqual(
                checker.check("final_answer", {"answer": "x"}).decision, "ALLOW"
            )

    def test_t2_uuid_case_insensitive(self):
        checker = T2ExecutionCheckV2(FakeCrapiAdapter())
        result = checker.check("get_vehicle_location", {"vehicle_uuid": OWN.upper()})
        self.assertEqual(result.decision, "ALLOW")

    def test_t2_other_vehicle_denied(self):
        checker = T2ExecutionCheckV2(FakeCrapiAdapter())
        result = checker.check("get_vehicle_location", {"vehicle_uuid": OTHER})
        self.assertEqual(result.decision, "DENY")

    def test_unknown_tool_denied(self):
        checker = T2BExecutionCheckV2(7)
        self.assertEqual(checker.check("delete_basket", {}).decision, "DENY")

    def test_t2b_other_basket_denied(self):
        checker = T2BExecutionCheckV2(7)
        result = checker.check("get_basket_contents", {"basket_id": 6})
        self.assertEqual(result.decision, "DENY")


class TestJuiceShopAdapterV2(unittest.TestCase):
    def test_ownership_without_reading_target(self):
        session = FakeSession(own_basket([(1, 11), (2, 12)]))
        adapter = JuiceShopAdapterV2("http://shop", session, 7)

        self.assertTrue(adapter.is_my_basket_item(11))
        self.assertFalse(adapter.is_my_basket_item(9))
        self.assertFalse(any("/api/BasketItems/" in u for u in session.urls))

    def test_t4_check_with_adapter_v2(self):
        session = FakeSession(own_basket([(1, 11)]))
        checker = T4ExecutionCheckV2(JuiceShopAdapterV2("http://shop", session, 7))

        self.assertEqual(
            checker.check("update_basket_item", {"item_id": 11, "quantity": 2}).decision,
            "ALLOW",
        )
        self.assertEqual(
            checker.check("update_basket_item", {"item_id": 9, "quantity": 2}).decision,
            "DENY",
        )


class TestMediation(unittest.TestCase):
    def test_uncovered_tool_rejected(self):
        checker = T2BExecutionCheckV2(7)
        agent = fake_guarded_agent(
            checker, ["get_basket_contents", "raw_http_get", "final_answer"], None
        )
        with self.assertRaises(RuntimeError):
            assert_complete_mediation(agent, checker)

    def test_covered_tools_accepted(self):
        checker = T2BExecutionCheckV2(7)
        agent = fake_guarded_agent(
            checker, ["get_my_basket_id", "get_basket_contents", "final_answer"], None
        )
        assert_complete_mediation(agent, checker)


class TestLayeredOrchestrator(unittest.TestCase):
    TOOLS = ["get_my_basket_id", "get_basket_contents", "final_answer"]

    def make(self, behavior):
        return fake_guarded_agent(T2BExecutionCheckV2(7), self.TOOLS, behavior)

    def test_input_deny_skips_agent(self):
        calls = []
        agent = self.make(lambda task: calls.append(task) or "x")
        result = LayeredOrchestrator(FixedChecker("DENY")).run("t", agent)

        self.assertEqual((result.status, result.blocked_by), ("BLOCKED", "input"))
        self.assertFalse(result.agent_executed)
        self.assertEqual(calls, [])

    def test_input_review_runs_agent(self):
        agent = self.make(lambda task: "basket contents")
        result = LayeredOrchestrator(FixedChecker("REVIEW")).run("t", agent)

        self.assertEqual(result.status, "ALLOWED")
        self.assertEqual(result.input_decision, "REVIEW")

    def test_input_allow_still_guarded(self):
        def behavior(task):
            raise SafeAIExecutionDenied("denied")

        result = LayeredOrchestrator(FixedChecker("ALLOW")).run("t", self.make(behavior))
        self.assertEqual((result.status, result.blocked_by), ("BLOCKED", "execution"))

    def test_output_review_does_not_block(self):
        result = LayeredOrchestrator(
            FixedChecker("REVIEW"), FixedChecker("REVIEW")
        ).run("t", self.make(lambda task: "ok"))
        self.assertEqual(result.status, "ALLOWED")

    def test_output_deny_blocks(self):
        result = LayeredOrchestrator(
            FixedChecker("ALLOW"), FixedChecker("DENY")
        ).run("t", self.make(lambda task: "ok"))
        self.assertEqual((result.status, result.blocked_by), ("BLOCKED", "output"))

    def test_unguarded_agent_rejected(self):
        with self.assertRaises(TypeError):
            LayeredOrchestrator(FixedChecker("ALLOW")).run("t", object())


if __name__ == "__main__":
    unittest.main()
