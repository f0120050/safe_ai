class CrapiAdapter:
    def __init__(self, base_url, session):
        self.base_url = base_url.rstrip("/")
        self.session = session

    def get_authorized_vehicle_uuids(self):
        """Retrieve authorized vehicle UUIDs for the current session."""

        response = self.session.get(
            f"{self.base_url}/identity/api/v2/vehicle/vehicles",
            timeout=15
        )

        response.raise_for_status()
        vehicles = response.json()

        if not isinstance(vehicles, list):
            raise ValueError("Invalid vehicle list response")

        uuids = set()

        for vehicle in vehicles:
            if not isinstance(vehicle, dict):
                raise ValueError("Invalid vehicle record")

            uuid = vehicle.get("uuid")

            if not isinstance(uuid, str) or not uuid:
                raise ValueError("Invalid vehicle UUID")

            uuids.add(uuid)

        return uuids
