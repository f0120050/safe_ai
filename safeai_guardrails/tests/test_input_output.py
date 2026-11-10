import unittest
from unittest.mock import Mock

from safeai_guardrails.checks.input_check import InputSafetyCheck
from safeai_guardrails.checks.output_check import OutputSafetyCheck
from safeai_guardrails.orchestrator import (
    SafeAIOrchestrator,
    AgentExecutionResult,
)


USER_Y_UUID = "08ca470e-dee0-4419-8671-2cc7e16bc44d"
USER_B_UUID = "7919479f-8f90-4f07-a786-07721111eafd"


class FakeCrapiAdapter:
    def get_authorized_vehicle_uuids(self):
        return {USER_Y_UUID}


class TestInputOutputIntegration(unittest.TestCase):

    def setUp(self):
        adapter = FakeCrapiAdapter()

        self.guard = SafeAIOrchestrator(
            input_checker=InputSafetyCheck(adapter),
            output_checker=OutputSafetyCheck(adapter),
        )

    def test_authorized_input_and_output(self):
        agent = Mock(return_value=AgentExecutionResult(
            response=f"Vehicle {USER_Y_UUID} location retrieved.",
            provenance={
                "verified": True,
                "resource_ids": [USER_Y_UUID],
            },
        ))

        result = self.guard.run(
            f"Get location of vehicle {USER_Y_UUID}", agent
        )

        self.assertEqual(result.status, "ALLOWED")
        self.assertTrue(result.agent_executed)
        agent.assert_called_once()

    def test_authorized_input_unauthorized_output(self):
        agent = Mock(return_value=AgentExecutionResult(
            response="Vehicle location: 37.746880, -84.301460",
            provenance={
                "verified": True,
                "resource_ids": [USER_B_UUID],
            },
        ))

        result = self.guard.run(
            f"Get location of vehicle {USER_Y_UUID}", agent
        )

        self.assertEqual(result.status, "BLOCKED")
        self.assertTrue(result.agent_executed)
        self.assertIsNone(result.response)

    def test_unauthorized_input_blocks_agent(self):
        agent = Mock(return_value="Should never execute")

        result = self.guard.run(
            f"Get location of vehicle {USER_B_UUID}", agent
        )

        self.assertEqual(result.status, "BLOCKED")
        self.assertFalse(result.agent_executed)
        agent.assert_not_called()

    def test_missing_provenance_blocks_output(self):
        agent = Mock(return_value="Vehicle location retrieved.")

        result = self.guard.run(
            f"Get location of vehicle {USER_Y_UUID}", agent
        )

        self.assertEqual(result.status, "BLOCKED")
        self.assertTrue(result.agent_executed)
        self.assertIsNone(result.response)


if __name__ == "__main__":
    unittest.main()
