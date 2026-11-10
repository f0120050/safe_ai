"""Sequential SafeAI batch runner with isolated workers and timeouts."""

import argparse
import getpass
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from safeai_experiments.config.experiment_config import (
    ATTACKS,
    MODELS,
    SAFETY_MODES,
)
from safeai_experiments.config.result_schema import ExperimentResult
from safeai_experiments.runners.result_writer import ResultWriter
from safeai_experiments.runners.audit_writer import AuditWriter


RESULTS_PATH = Path("safeai_experiments/results/results.jsonl")
WORKER_MODULE = "safeai_experiments.runners.experiment_runner"


def record_exists(run_id, results_path=None):
    """Check whether the worker already saved its result."""
    path = Path(results_path) if results_path is not None else RESULTS_PATH
    if not path.exists():
        return False

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("run_id") == run_id:
                return True

    return False


def save_failure(
    run_id, model, mode, attack, repetition,
    status, elapsed, error_type,
    results_dir=None, pair_id=None,
):
    """Persist an unknown-outcome technical failure."""
    results_path = (Path(results_dir) / "results.jsonl") if results_dir else RESULTS_PATH
    if record_exists(run_id, results_path):
        print("Worker result already saved; skipping fallback record.")
        return

    result = ExperimentResult(
        run_id=run_id,
        pair_id=pair_id or f"{attack}-{model}-{repetition:03d}",
        model=MODELS[model],
        attack_id=attack,
        safety_mode=mode,
        repetition=repetition,
        status=status,
    )

    # Unknown, rather than false: the worker may have partially executed.
    result.agent_executed = None
    result.unauthorized_http_access = None
    result.unauthorized_final_disclosure = None
    result.task_completed = None
    result.agent_step_count = None
    result.json_error_count = None
    result.tool_call_error_count = None
    result.timeout_occurred = status == "TIMEOUT"
    result.total_time_s = elapsed
    result.error_type = error_type
    result.error_message = "Worker did not complete normally."

    if results_dir:
        ResultWriter(results_dir=results_dir).save(result)
        AuditWriter(directory=Path(results_dir) / "logs").save(result, [])
    else:
        ResultWriter().save(result)
        AuditWriter().save(result, [])
    print(f"Recorded technical failure: {run_id} ({status})")


def terminate_process_group(process):
    """Stop the worker and its child processes on Linux."""
    if process.poll() is not None:
        return

    os.killpg(process.pid, signal.SIGTERM)

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_worker(model, mode, attack, repetition, password, timeout_s,
               results_dir=None, pair_prefix=None):
    run_id = f"real-{uuid4().hex}"

    command = [
        sys.executable,
        "-m", WORKER_MODULE,
        "--model", model,
        "--mode", mode,
        "--attack", attack,
        "--repetition", str(repetition),
    ]

    # Only the file descriptor number is placed in the environment.
    # The password itself is never placed in environment variables.
    read_fd, write_fd = os.pipe()

    environment = os.environ.copy()
    environment["SAFEAI_PASSWORD_FD"] = str(read_fd)
    environment["SAFEAI_RUN_ID"] = run_id

    pair_id = (
        f"{pair_prefix}-{attack}-{model}-{repetition:03d}"
        if pair_prefix else f"{attack}-{model}-{repetition:03d}"
    )
    environment["SAFEAI_PAIR_ID"] = pair_id
    if results_dir is not None:
        environment["SAFEAI_RESULTS_DIR"] = str(results_dir)

    results_path = (
        Path(results_dir) / "results.jsonl"
        if results_dir is not None else RESULTS_PATH
    )

    started = time.perf_counter()
    process = None
    status = None
    error_type = None

    try:
        process = subprocess.Popen(
            command,
            env=environment,
            pass_fds=(read_fd,),
            start_new_session=True,
        )

        os.close(read_fd)
        read_fd = None

        # Password travels only through the temporary OS pipe.
        with os.fdopen(write_fd, "wb") as pipe:
            write_fd = None
            pipe.write(password.encode("utf-8") + b"\n")
            pipe.flush()

        try:
            return_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            error_type = "BatchTimeout"
            terminate_process_group(process)
        else:
            if return_code != 0:
                status = "TECHNICAL_ERROR"
                error_type = "WorkerExitError"

        elapsed = time.perf_counter() - started

        if not record_exists(run_id, results_path):
            if status is None:
                status = "TECHNICAL_ERROR"
                error_type = "MissingResultRecord"

            save_failure(
                run_id, model, mode, attack, repetition,
                status, elapsed, error_type,
                results_dir=results_dir, pair_id=pair_id,
            )

        print(f"Batch run completed: {run_id}")
        print(f"Elapsed: {elapsed:.2f}s")

        return status is None and record_exists(run_id, results_path)

    except Exception as exc:
        if process is not None:
            terminate_process_group(process)

        elapsed = time.perf_counter() - started

        save_failure(
            run_id, model, mode, attack, repetition,
            "TECHNICAL_ERROR", elapsed, type(exc).__name__,
            results_dir=results_dir, pair_id=pair_id,
        )
        print("Batch infrastructure error:", type(exc).__name__)
        return False

    finally:
        if read_fd is not None:
            os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=MODELS, default="qwen")
    parser.add_argument(
        "--mode", choices=SAFETY_MODES, default="input_only"
    )
    parser.add_argument("--attack", choices=ATTACKS, default="T2")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--start-repetition", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--pair-prefix", default=None)

    args = parser.parse_args()

    if args.repeats < 1 or args.start_repetition < 1:
        parser.error("Repeat counts and starting repetition must be positive.")

    if args.timeout < 10:
        parser.error("Timeout must be at least 10 seconds.")

    # Ensure the worker supports parent-assigned run identifiers.
    worker_path = Path(
        "safeai_experiments/runners/experiment_runner.py"
    )
    if "SAFEAI_RUN_ID" not in worker_path.read_text():
        parser.error("Worker does not support SAFEAI_RUN_ID.")

    password = getpass.getpass("User Y password (once): ")

    successes = 0

    for index in range(args.repeats):
        repetition = args.start_repetition + index

        print(
            f"\n=== Batch {index + 1}/{args.repeats}"
            f" | {args.model} | {args.mode}"
            f" | repetition {repetition} ===",
            flush=True,
        )

        completed = run_worker(
            model=args.model,
            mode=args.mode,
            attack=args.attack,
            repetition=repetition,
            password=password,
            timeout_s=args.timeout,
            results_dir=args.results_dir,
            pair_prefix=args.pair_prefix,
        )

        successes += int(completed)

    print("\nBATCH FINISHED")
    print("Runs requested:", args.repeats)
    print("Runs with worker records:", successes)
    print("Other runs: inspect TIMEOUT / TECHNICAL_ERROR records.")


if __name__ == "__main__":
    main()
