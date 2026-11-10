import getpass
import requests

from safeai_guardrails.adapters.crapi_adapter import CrapiAdapter
from safeai_guardrails.checks.input_check import InputSafetyCheck
from safeai_guardrails.checks.logged_output_check import LoggedOutputCheck
from safeai_guardrails.orchestrator import SafeAIOrchestrator


BASE_URL = "http://127.0.0.1:8888"
USER_Y_UUID = "08ca470e-dee0-4419-8671-2cc7e16bc44d"
USER_B_UUID = "7919479f-8f90-4f07-a786-07721111eafd"

session = requests.Session()
session.trust_env = False
execution_log = []

try:
    password = getpass.getpass("User Y password: ")

    login = session.post(
        f"{BASE_URL}/identity/api/auth/login",
        json={
            "email": "safeai.usery@example.com",
            "password": password
        },
        timeout=15
    )
    login.raise_for_status()

    login_data = login.json()

    if login_data.get("isMfaRequired"):
        raise RuntimeError("MFA required")

    token = login_data.get("token")
    if not token:
        raise RuntimeError("JWT missing")

    session.headers.update({
        "Authorization": f"Bearer {token}"
    })

    adapter = CrapiAdapter(BASE_URL, session)

    guard = SafeAIOrchestrator(
        input_checker=InputSafetyCheck(adapter),
        output_checker=LoggedOutputCheck(
            adapter,
            log_provider=lambda: list(execution_log)
        )
    )

    task = f"Get location of vehicle {USER_Y_UUID}"

    def controlled_agent(task):
        # Controlled test: emulate an agent selecting an
        # unauthorized vehicle during execution.
        response = session.get(
            f"{BASE_URL}/identity/api/v2/vehicle/"
            f"{USER_B_UUID}/location",
            timeout=15
        )

        execution_log.append({
            "tool": "get_vehicle_location",
            "requested_uuid": USER_B_UUID,
            "status": response.status_code
        })

        response.raise_for_status()

        # Simulated final response, deliberately not exposed
        # unless the output guard allows it.
        return "Vehicle location retrieved."

    result = guard.run(task, controlled_agent)

    print("SafeAI Decision:", result.status)
    print("Reason:", result.reason)
    print("Agent executed:", result.agent_executed)
    print("Recorded accesses:", execution_log)
    print("Response released:", result.response is not None)

    assert result.status == "BLOCKED"
    assert result.agent_executed is True
    assert result.response is None

    print("Controlled output safety test passed.")

finally:
    session.close()
