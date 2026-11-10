import requests


class SafetyController:
    def __init__(self, base_url, session):
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.decision_log = []

    def authorize_vehicle_location(self, vehicle_uuid):
        decision = "DENY"
        reason = "Ownership could not be verified"

        if not isinstance(vehicle_uuid, str) or not vehicle_uuid:
            reason = "Invalid vehicle UUID"
        else:
            try:
                response = self.session.get(
                    f"{self.base_url}/identity/api/v2/vehicle/vehicles",
                    timeout=15
                )
                response.raise_for_status()

                vehicles = response.json()

                if not isinstance(vehicles, list):
                    raise ValueError("Invalid vehicle list response")

                authorized_uuids = {
                    vehicle["uuid"]
                    for vehicle in vehicles
                }

                if vehicle_uuid in authorized_uuids:
                    decision = "ALLOW"
                    reason = "Vehicle ownership verified"
                else:
                    reason = "Vehicle not owned by authenticated user"

            except (
                requests.RequestException,
                ValueError,
                KeyError,
                TypeError
            ):
                reason = "Ownership verification failed"

        self.decision_log.append({
            "vehicle_uuid": vehicle_uuid,
            "decision": decision,
            "reason": reason
        })

        return decision == "ALLOW"
