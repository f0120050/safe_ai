"""Isolated, thermally guarded scheduler for legitimate-task experiments.

- Each run executes in its own process with a hard wall-clock limit.
- Before each run: pause while GPU is hot, stop if RAM is low.
- During each run: abort the run if the GPU reaches a critical temperature.
- Aborted runs are recorded and re-run on the next invocation.
"""

import argparse
import getpass
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from uuid import uuid4

from safeai_experiments.config.experiment_config import MODELS
from safeai_experiments.config.legit_tasks import T2_LEGIT_TASKS
from safeai_experiments.runners.legit_t2_worker import (
    MODES as T2_MODES,
    RESULTS_ROOT as T2_ROOT,
    VALID,
    completed_keys,
)
from safeai_experiments.runners.ollama_model_manager import (
    gpu_ready, prepare_model, stop_all_models,
)


SCENARIOS = {
    "T2": {
        "worker": "safeai_experiments.runners.legit_t2_worker",
        "tasks": T2_LEGIT_TASKS,
        "modes": T2_MODES,
        "root": T2_ROOT,
    },
}


def gpu_temperature():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def mem_available_gb():
    with open("/proc/meminfo", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    return float("inf")


def wait_for_safe_conditions(args):
    ram = mem_available_gb()
    if ram < args.min_ram_gb:
        raise RuntimeError(f"Low RAM: {ram:.1f} GB available; stopping")

    temp = gpu_temperature()
    if temp is None:
        print("Warning: GPU temperature unavailable", flush=True)
        return
    if temp >= args.max_temp:
        print(f"GPU {temp} C >= {args.max_temp} C; pausing until "
              f"<= {args.resume_temp} C", flush=True)
        while (gpu_temperature() or 0) > args.resume_temp:
            time.sleep(15)
        print(f"GPU cooled to {gpu_temperature()} C; continuing", flush=True)


def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def run_isolated(command, password, timeout, critical_temp):
    read_fd, write_fd = os.pipe()
    env = os.environ.copy()
    env["SAFEAI_PASSWORD_FD"] = str(read_fd)

    process = subprocess.Popen(
        command, env=env, pass_fds=(read_fd,), start_new_session=True,
    )
    os.close(read_fd)
    with os.fdopen(write_fd, "wb") as stream:
        stream.write(password.encode("utf-8") + b"\n")

    started = time.perf_counter()
    while True:
        try:
            process.wait(timeout=5)
            return "DONE", time.perf_counter() - started
        except subprocess.TimeoutExpired:
            pass

        elapsed = time.perf_counter() - started
        if elapsed > timeout:
            kill_group(process)
            return "TIMEOUT", elapsed

        temp = gpu_temperature()
        if temp is not None and temp >= critical_temp:
            kill_group(process)
            print(f"GPU reached {temp} C; run aborted", flush=True)
            return "THERMAL_ABORT", elapsed


def append_abort_record(path, args, rep, mode, task, status, elapsed):
    record = {
        "run_id": f"legit-abort-{uuid4().hex}",
        "scenario": args.scenario,
        "task_id": task["id"],
        "category": task["category"],
        "model": MODELS[args.model],
        "mode": mode,
        "repetition": rep,
        "status": status,
        "error_type": "SchedulerAbort",
        "total_time_s": elapsed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def summarize(path, modes):
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    latest = {}
    for r in lines:
        key = (r["repetition"], r["mode"], r["task_id"])
        if r.get("status") in VALID or key not in latest:
            latest[key] = r
    records = list(latest.values())

    print("\nSUMMARY (completed / blocked / not-valid / total)")
    for mode in modes:
        rows = [r for r in records if r["mode"] == mode]
        ok = sum(r.get("legit_task_completed") is True for r in rows)
        blocked = sum(r["status"] == "BLOCKED" for r in rows)
        bad = sum(r["status"] not in VALID for r in rows)
        by = Counter(r.get("blocked_by") for r in rows if r.get("blocked_by"))
        print(f"  {mode:18s} {ok}/{blocked}/{bad}/{len(rows)}  blocked_by={dict(by)}")

    print("Non-valid attempts in file:",
          dict(Counter(r["status"] for r in lines if r["status"] not in VALID)))
    print("Unauthorized HTTP in legitimate runs:",
          sum(r.get("unauthorized_http_access") is True for r in records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="T2", choices=SCENARIOS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", required=True)
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--run-timeout", type=int, default=300)
    parser.add_argument("--max-temp", type=int, default=85)
    parser.add_argument("--resume-temp", type=int, default=70)
    parser.add_argument("--critical-temp", type=int, default=92)
    parser.add_argument("--min-ram-gb", type=float, default=4.0)
    parser.add_argument("--cooldown", type=float, default=5.0)
    args = parser.parse_args()

    cfg = SCENARIOS[args.scenario]
    modes = args.modes or list(cfg["modes"])
    out_dir = cfg["root"] / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    password = getpass.getpass("Password (once): ")
    target = MODELS[args.model].removeprefix("ollama_chat/")

    print("Unloading all Ollama models...", flush=True)
    stop_all_models()
    prep = prepare_model(target, minimum_gpu_percent=80)
    print("GPU:", prep["reason"], flush=True)
    if not prep["gpu_ready"]:
        sys.exit("Insufficient GPU allocation")

    plan = []
    for rep in range(1, args.repeats + 1):
        offset = (rep - 1) % len(modes)
        for mode in modes[offset:] + modes[:offset]:
            for task in cfg["tasks"]:
                plan.append((rep, mode, task))

    try:
        for rep, mode, task in plan:
            if (rep, mode, task["id"]) in completed_keys(results_path):
                continue

            wait_for_safe_conditions(args)

            ready, reason = gpu_ready(target)
            if not ready:
                prep = prepare_model(target, minimum_gpu_percent=80)
                if not prep["gpu_ready"]:
                    raise RuntimeError(f"GPU residency lost: {prep['reason']}")

            print(f"\n=== {args.model} {mode} {task['id']} rep={rep} "
                  f"(GPU {gpu_temperature()} C) ===", flush=True)

            command = [
                sys.executable, "-m", cfg["worker"],
                "--model", args.model, "--repeats", str(args.repeats),
                "--label", args.label, "--resume", "--skip-gpu-prep",
                "--no-summary", "--run-timeout", str(args.run_timeout),
                "--only-rep", str(rep), "--only-mode", mode,
                "--only-task", task["id"],
            ]
            outcome, elapsed = run_isolated(
                command, password,
                timeout=args.run_timeout + 90,
                critical_temp=args.critical_temp,
            )
            if outcome != "DONE":
                append_abort_record(results_path, args, rep, mode, task,
                                    outcome, elapsed)
                print(f"Run recorded as {outcome}", flush=True)

            time.sleep(args.cooldown)
    finally:
        print("Unloading model:", target, flush=True)
        subprocess.run(["ollama", "stop", target], check=False, timeout=30)

    summarize(results_path, modes)


if __name__ == "__main__":
    main()
