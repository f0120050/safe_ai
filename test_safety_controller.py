import getpass
import json
import requests

from safety_controller import SafetyController

BASE_URL = "http://127.0.0.1:8888"
USER_Y_EMAIL = "safeai.usery@example.com"

USER_Y_UUID = "08ca470e-dee0-4419-8671-2cc7e16bc44d"
USER_B_UUID = "7919479f-8f90-4f07-a786-07721111eafd"

session = requests.Session()
session.trust_env = False

password = getpass.getpass("User Y password: ")

response = session.post(
    f"{BASE_URL}/identity/api/auth/login",
    json={
        "email": USER_Y_EMAIL,
        "password": password
    },
    timeout=15
)
response.raise_for_status()

token = response.json()["token"]
session.headers.update({
    "Authorization": f"Bearer {token}"
})

controller = SafetyController(BASE_URL, session)

for label, uuid in [
    ("User Y", USER_Y_UUID),
    ("User B", USER_B_UUID)
]:
    allowed = controller.authorize_vehicle_location(uuid)
    print(f"{label}: {'ALLOW' if allowed else 'DENY'}")

print("\nSAFETY CONTROLLER LOG:")
print(json.dumps(controller.decision_log, indent=2))
