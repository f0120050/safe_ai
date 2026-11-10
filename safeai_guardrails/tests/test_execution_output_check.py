import unittest

from safeai_guardrails.checks.execution_output_check import ExecutionOutputCheck


USER_Y_UUID = "08ca470e-dee0-4419-8671-2cc7e16bc44d"
USER_B_UUID = "7919479f-8f90-4f07-a786-07721111eafd"


class FakeCrapiAdapter:
    def get_authorized_vehicle_uuids(self):
        return {USER_Y_UUID}


class FailingCrapiAdapter:
    def get_authorized_vehicle_uuids(self):
        raise ConnectionError("Authorization API unavailable")


class TestExecutionOutputCheck(unittest.TestCase):

    def setUp(self):
        self.checker = ExecutionOutputCheck(FakeCrapiAdapter())

    def test_authorized_access(self):
        log = [{
            "tool": "get_vehicle_location",
            "requested_uuid": USER_Y_UUID,
            "status": 200,
        }]

        result = self.checker.check("Location retrieved.", log)
        self.assertEqual(result.decision, "ALLOW")

    def test_unauthorized_access(self):
        log = [{
            "tool": "get_vehicle_location",
            "requested_uuid": USER_B_UUID,
            "status": 200,
        }]

        result = self.checker.check("Location retrieved.", log)
        self.assertEqual(result.decision, "DENY")

    def test_coordinates_without_uuid(self):
        log = [{
            "tool": "get_vehicle_location",
            "requested_uuid": USER_B_UUID,
            "status": 200,
        }]

        response = "Vehicle location: 37.746880, -84.301460"

        result = self.checker.check(response, log)
        self.assertEqual(result.decision, "DENY")

    def test_mixed_access(self):
        log = [
            {
                "tool": "get_vehicle_location",
                "requested_uuid": USER_Y_UUID,
                "status": 200,
            },
            {
                "tool": "get_vehicle_location",
                "requested_uuid": USER_B_UUID,
                "status": 200,
            },
        ]

        result = self.checker.check("Locations retrieved.", log)
        self.assertEqual(result.decision, "DENY")

    def test_missing_access_log(self):
        result = self.checker.check("Location retrieved.", [])
        self.assertEqual(result.decision, "REVIEW")

    def test_authorization_api_failure(self):
        checker = ExecutionOutputCheck(FailingCrapiAdapter())

        log = [{
            "tool": "get_vehicle_location",
            "requested_uuid": USER_Y_UUID,
            "status": 200,
        }]

        result = checker.check("Location retrieved.", log)
        self.assertEqual(result.decision, "DENY")


if __name__ == "__main__":
    unittest.main()
