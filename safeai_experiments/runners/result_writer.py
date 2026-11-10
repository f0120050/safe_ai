import csv
import json
import os
from pathlib import Path

from safeai_experiments.config.result_schema import ExperimentResult


class ResultWriter:
    def __init__(self, results_dir="safeai_experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.results_dir / "results.jsonl"
        self.csv_path = self.results_dir / "results.csv"

    def save(self, result: ExperimentResult):
        record = result.to_dict()

        # Read existing JSONL records to prevent duplicate run IDs.
        existing = []
        if self.jsonl_path.exists():
            with self.jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing.append(json.loads(line))

        if any(r.get("run_id") == result.run_id for r in existing):
            raise ValueError(
                f"Duplicate run_id: {result.run_id}"
            )

        # Persist JSONL immediately.
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(result.to_json() + "\n")
            f.flush()
            os.fsync(f.fileno())

        # Rebuild CSV from the complete JSONL archive.
        existing.append(record)

        fieldnames = list(ExperimentResult.__dataclass_fields__.keys())

        temp_path = self.csv_path.with_suffix(".csv.tmp")

        with temp_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(existing)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, self.csv_path)

        return self.jsonl_path, self.csv_path
