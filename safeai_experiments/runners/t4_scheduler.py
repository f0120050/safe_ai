"""T4 main experiment scheduler."""

import argparse
import getpass
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from safeai_experiments.config.experiment_config import MODELS
from safeai_experiments.runners.batch_runner import save_failure
from safeai_experiments.runners.ollama_model_manager import (
    gpu_ready, prepare_model, stop_all_models,
)

MODES = ("prompt_only", "input_only", "execution_only")
ROOT = Path("safeai_experiments/main_results/T4")
WORKER = "safeai_experiments.runners.t4_worker"


def records(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


def run_worker(model, mode, repetition, passwords, timeout, directory):
    run_id = f"real-{uuid4().hex}"
    pair_id = f"main-T4-{model}-{repetition:03d}"
    result_path = directory / "results.jsonl"

    pipes = [os.pipe(), os.pipe()]
    env = os.environ.copy()
    env.update({
        "SAFEAI_PASSWORD_FD": str(pipes[0][0]),
        "SAFEAI_TARGET_PASSWORD_FD": str(pipes[1][0]),
        "SAFEAI_RUN_ID": run_id,
        "SAFEAI_PAIR_ID": pair_id,
        "SAFEAI_RESULTS_DIR": str(directory),
    })

    process = None
    status = None
    error_type = None

    try:
        process = subprocess.Popen(
            [
                sys.executable, "-m", WORKER,
                "--model", model,
                "--mode", mode,
                "--repetition", str(repetition),
            ],
            env=env,
            pass_fds=tuple(p[0] for p in pipes),
            start_new_session=True,
        )

        for (read_fd, write_fd), password in zip(pipes, passwords):
            os.close(read_fd)
            with os.fdopen(write_fd, "wb") as stream:
                stream.write(password.encode("utf-8") + b"\n")

        pipes = []

        try:
            return_code = process.wait(timeout=timeout)
            if return_code != 0:
                status = "TECHNICAL_ERROR"
                error_type = "WorkerExitError"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            error_type = "BatchTimeout"
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    except Exception as exc:
        status = "TECHNICAL_ERROR"
        error_type = type(exc).__name__
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    finally:
        for read_fd, write_fd in pipes:
            for fd in (read_fd, write_fd):
                try:
                    os.close(fd)
                except OSError:
                    pass

    matches = [
        r for r in records(result_path)
        if r.get("run_id") == run_id
    ]

    if not matches:
        save_failure(
            run_id=run_id,
            model=model,
            mode=mode,
            attack="T4",
            repetition=repetition,
            status=status or "TECHNICAL_ERROR",
            elapsed=0,
            error_type=error_type or "MissingResultRecord",
            results_dir=str(directory),
            pair_id=pair_id,
        )
        return False

    outcome = matches[-1]["status"]
    print("Recorded status:", outcome, flush=True)
    return status is None and outcome in ("ALLOWED", "BLOCKED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1 or args.timeout < 10:
        parser.error("Invalid repetition count or timeout")

    directory = ROOT / args.model
    path = directory / "results.jsonl"

    completed = {
        (r.get("pair_id"), r.get("safety_mode"))
        for r in records(path)
        if r.get("status") in ("ALLOWED", "BLOCKED")
    }

    plan = []
    for rep in range(1, args.repeats + 1):
        offset = (rep - 1) % len(MODES)
        order = MODES[offset:] + MODES[:offset]
        for mode in order:
            pair = f"main-T4-{args.model}-{rep:03d}"
            if (pair, mode) not in completed:
                plan.append((rep, mode))

    print("Model:", args.model)
    print("Total planned:", args.repeats * len(MODES))
    print("Completed:", args.repeats * len(MODES) - len(plan))
    print("Pending:", len(plan))
    print("Results:", directory, flush=True)

    if args.dry_run or not plan:
        return

    target = MODELS[args.model].removeprefix("ollama_chat/")

    try:
        stop_all_models()
        preparation = prepare_model(
            target, minimum_gpu_percent=80
        )
        print("GPU:", preparation["reason"], flush=True)

        if not preparation["gpu_ready"]:
            raise RuntimeError("Insufficient GPU allocation")

        passwords = (
            getpass.getpass("User A password (once): "),
            getpass.getpass("User B password (once): "),
        )

        for index, (rep, mode) in enumerate(plan, 1):
            ready, reason = gpu_ready(target)
            if not ready:
                raise RuntimeError(f"GPU unavailable: {reason}")

            print(
                f"\n=== {index}/{len(plan)} "
                f"{args.model} {mode} repetition={rep} ===",
                flush=True,
            )

            successful = run_worker(
                args.model, mode, rep, passwords,
                args.timeout, directory,
            )

            if not successful:
                print(
                    "Technical failure recorded; continuing "
                    "with the next experiment.",
                    flush=True,
                )

    finally:
        subprocess.run(
            ["ollama", "stop", target],
            check=False,
            timeout=30,
        )

    print("T4 MAIN EXPERIMENT BATCH FINISHED")


if __name__ == "__main__":
    main()
