import unittest
from unittest.mock import Mock

from safeai_guardrails.checks.input_check import InputSafetyCheck
from safeai_guardrails.checks.logged_output_check import LoggedOutputCheck
from safeai_guardrails.orchestrator import SafeAIOrchestrator


USER_Y_UUID = "08ca470e-dee0-4419-8671-2cc7e16bc44d"
USER_B_UUID = "7919479f-8f90-4f07-a786-07721111eafd"


class FakeCrapiAdapter:
    def get_authorized_vehicle_uuids(self):
        return {USER_Y_UUID}


class TestLoggedOrchestrator(unittest.TestCase):

    def setUp(self):
        adapter = FakeCrapiAdapter()
        self.execution_log = []

        self.guard = SafeAIOrchestrator(
            input_checker=InputSafetyCheck(adapter),
            output_checker=LoggedOutputCheck(
                adapter,
                log_provider=lambda: list(self.execution_log),
            ),
        )

    def test_authorized_execution(self):
        def agent(task):
            self.execution_log.append({
                "tool": "get_vehicle_location",
                "requested_uuid": USER_Y_UUID,
                "status": 200,
            })
            return "Vehicle location retrieved."

        result = self.guard.run(
            f"Get location of vehicle {USER_Y_UUID}",
            agent,
        )

        self.assertEqual(result.status, "ALLOWED")
        self.assertTrue(result.agent_executed)

    def test_unauthorized_execution_after_safe_input(self):
        def agent(task):
            self.execution_log.append({
                "tool": "get_vehicle_location",
                "requested_uuid": USER_B_UUID,
                "status": 200,
            })
            return "Location: 37.746880, -84.301460"

        result = self.guard.run(
            f"Get location of vehicle {USER_Y_UUID}",
            agent,
        )

        self.assertEqual(result.status, "BLOCKED")
        self.assertTrue(result.agent_executed)
        self.assertIsNone(result.response)

    def test_unauthorized_input_never_runs_agent(self):
        agent = Mock(return_value="Should not execute")

        result = self.guard.run(
            f"Get location of vehicle {USER_B_UUID}",
            agent,
        )

        self.assertEqual(result.status, "BLOCKED")
        self.assertFalse(result.agent_executed)
        agent.assert_not_called()

    def test_unauthorized_uuid_in_final_output(self):
        def agent(task):
            self.execution_log.append({
                "tool": "get_vehicle_location",
                "requested_uuid": USER_Y_UUID,
                "status": 200,
            })
            return f"Vehicle {USER_B_UUID} location retrieved."

        result = self.guard.run(
            f"Get location of vehicle {USER_Y_UUID}",
            agent,
        )

        self.assertEqual(result.status, "BLOCKED")
        self.assertTrue(result.agent_executed)
        self.assertIsNone(result.response)


if __name__ == "__main__":
    unittest.main()
