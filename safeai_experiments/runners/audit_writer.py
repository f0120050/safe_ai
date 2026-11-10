"""Write a minimal redacted audit record for each experiment."""

import json
import os
from pathlib import Path


class AuditWriter:
    def __init__(self, directory="safeai_experiments/logs"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, result, http_log):
        events = []

        for entry in http_log:
            events.append({
                "tool": entry.get("tool"),
                "http_status": entry.get("status"),
                "location_request": (
                    entry.get("tool") == "get_vehicle_location"
                ),
                # Do not persist vehicle UUIDs or coordinates.
            })

        record = {
            "run_id": result.run_id,
            "pair_id": result.pair_id,
            "model": result.model,
            "attack_id": result.attack_id,
            "safety_mode": result.safety_mode,
            "status": result.status,
            "input_decision": result.input_decision,
            "output_decision": result.output_decision,
            "unauthorized_http_access": result.unauthorized_http_access,
            "unauthorized_final_disclosure": (
                result.unauthorized_final_disclosure
            ),
            "events": events,
        }

        destination = self.directory / f"{result.run_id}.json"
        temporary = destination.with_suffix(".json.tmp")

        with temporary.open("w", encoding="utf-8") as file:
            json.dump(record, file, indent=2)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary, destination)
        return destination
