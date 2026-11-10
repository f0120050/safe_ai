import unittest

from safeai_guardrails.checks.output_check import OutputSafetyCheck


USER_Y_UUID = "08ca470e-dee0-4419-8671-2cc7e16bc44d"
USER_B_UUID = "7919479f-8f90-4f07-a786-07721111eafd"


class FakeCrapiAdapter:
    def get_authorized_vehicle_uuids(self):
        return {USER_Y_UUID}


class FailingCrapiAdapter:
    def get_authorized_vehicle_uuids(self):
        raise ConnectionError("Authorization API unavailable")


class TestOutputSafetyCheck(unittest.TestCase):

    def setUp(self):
        self.checker = OutputSafetyCheck(FakeCrapiAdapter())

    def test_authorized_output_with_provenance(self):
        response = f"Vehicle {USER_Y_UUID} location retrieved."

        provenance = {
            "verified": True,
            "resource_ids": [USER_Y_UUID]
        }

        result = self.checker.check(response, provenance)
        self.assertEqual(result.decision, "ALLOW")

    def test_unauthorized_output(self):
        response = f"Vehicle {USER_B_UUID} location retrieved."

        provenance = {
            "verified": True,
            "resource_ids": [USER_B_UUID]
        }

        result = self.checker.check(response, provenance)
        self.assertEqual(result.decision, "DENY")

    def test_output_without_provenance(self):
        response = "Vehicle location: 37.746880, -84.301460"

        result = self.checker.check(response)
        self.assertEqual(result.decision, "REVIEW")

    def test_unauthorized_provenance_without_uuid_in_text(self):
        response = "Vehicle location: 37.746880, -84.301460"

        provenance = {
            "verified": True,
            "resource_ids": [USER_B_UUID]
        }

        result = self.checker.check(response, provenance)
        self.assertEqual(result.decision, "DENY")

    def test_invalid_provenance(self):
        response = "Vehicle location retrieved."

        provenance = {
            "verified": False,
            "resource_ids": [USER_Y_UUID]
        }

        result = self.checker.check(response, provenance)
        self.assertEqual(result.decision, "DENY")

    def test_authorization_api_failure(self):
        checker = OutputSafetyCheck(FailingCrapiAdapter())

        response = f"Vehicle {USER_Y_UUID} location retrieved."

        provenance = {
            "verified": True,
            "resource_ids": [USER_Y_UUID]
        }

        result = checker.check(response, provenance)
        self.assertEqual(result.decision, "DENY")


if __name__ == "__main__":
    unittest.main()
