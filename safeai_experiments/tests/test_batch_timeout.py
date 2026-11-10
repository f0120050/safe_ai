"""Isolated integration test for Batch Runner timeout handling."""

import json
from pathlib import Path

import safeai_experiments.runners.batch_runner as batch
from safeai_experiments.runners.result_writer import ResultWriter
from safeai_experiments.runners.audit_writer import AuditWriter


TEST_DIR = Path("safeai_experiments/synthetic_results/timeout_smoke")
RESULTS_DIR = TEST_DIR / "results"
LOGS_DIR = TEST_DIR / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Keep the real experiment data completely separate.
batch.RESULTS_PATH = RESULTS_DIR / "results.jsonl"
batch.WORKER_MODULE = "safeai_experiments.tests.timeout_worker"

batch.ResultWriter = lambda: ResultWriter(results_dir=RESULTS_DIR)
batch.AuditWriter = lambda: AuditWriter(directory=LOGS_DIR)

# Refuse to overwrite or mix previous test data.
if batch.RESULTS_PATH.exists():
    raise SystemExit(
        "Timeout test results already exist. "
        "Existing records were preserved."
    )

for repetition in (1, 2):
    print(f"\n=== Synthetic timeout test {repetition}/2 ===", flush=True)

    completed = batch.run_worker(
        model="qwen",
        mode="input_only",
        attack="T2",
        repetition=repetition,
        password="synthetic-test-only",
        timeout_s=1,
    )

    assert completed is False, "Expected a timeout, not completion."

records = [
    json.loads(line)
    for line in batch.RESULTS_PATH.read_text().splitlines()
    if line.strip()
]

assert len(records) == 2
assert all(r["status"] == "TIMEOUT" for r in records)
assert all(r["timeout_occurred"] is True for r in records)
assert all(r["unauthorized_http_access"] is None for r in records)
assert all(r["unauthorized_final_disclosure"] is None for r in records)
assert len(list(LOGS_DIR.glob("*.json"))) == 2

print("\nALL TIMEOUT TESTS PASSED")
print("Records:", len(records))
print("Results directory:", RESULTS_DIR)
print("Audit directory:", LOGS_DIR)
