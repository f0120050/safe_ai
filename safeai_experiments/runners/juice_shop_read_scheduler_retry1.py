"""T2-B main experiment scheduler for OWASP Juice Shop."""

import argparse
import getpass
import subprocess
from pathlib import Path

from safeai_experiments.config.experiment_config import (
    MODELS,
    SAFETY_MODES,
)
from safeai_experiments.runners import batch_runner
from safeai_experiments.runners.main_experiment_scheduler import (
    recorded_pairs,
)
from safeai_experiments.runners.ollama_model_manager import (
    gpu_ready,
    prepare_model,
    stop_all_models,
)


RESULTS_ROOT = Path("safeai_experiments/main_results_retry1")
WORKER_MODULE = "safeai_experiments.runners.juice_shop_read_worker"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1 or args.timeout < 10:
        parser.error("Invalid repeats or timeout")

    attack = "T2B"
    results_dir = RESULTS_ROOT / attack / args.model
    results_path = results_dir / "results.jsonl"

    completed = recorded_pairs(results_path)

    modes = list(SAFETY_MODES)
    plan = []

    for repetition in range(1, args.repeats + 1):
        offset = (repetition - 1) % len(modes)
        ordered = modes[offset:] + modes[:offset]

        for mode in ordered:
            plan.append((repetition, mode))

    pending = [
        (repetition, mode)
        for repetition, mode in plan
        if (
            f"main-T2B-{args.model}-{repetition:03d}",
            mode,
        ) not in completed
    ]

    print("Attack:", attack)
    print("Model:", args.model)
    print("Worker:", WORKER_MODULE)
    print("Planned:", len(plan))
    print("Recorded:", len(plan) - len(pending))
    print("Pending:", len(pending))
    print("Results:", results_dir)

    if args.dry_run or not pending:
        return

    # Route only this scheduler process to the Juice Shop worker.
    # The original batch_runner.py file remains unchanged.
    batch_runner.WORKER_MODULE = WORKER_MODULE

    target = MODELS[args.model].removeprefix("ollama_chat/")

    try:
        print("Unloading Ollama models...", flush=True)
        stop_all_models()

        preparation = prepare_model(
            target,
            minimum_gpu_percent=80,
        )

        print("GPU:", preparation["reason"], flush=True)

        if not preparation["gpu_ready"]:
            raise RuntimeError("Insufficient GPU allocation")

        password = getpass.getpass(
            "Juice Shop User A password (once): "
        )

        for index, (repetition, mode) in enumerate(pending, 1):
            ready, reason = gpu_ready(target)

            if not ready:
                raise RuntimeError(
                    f"GPU residency lost: {reason}"
                )

            print(
                f"\\n=== {index}/{len(pending)} "
                f"{args.model} {mode} "
                f"repetition={repetition} ===",
                flush=True,
            )

            batch_runner.run_worker(
                model=args.model,
                mode=mode,
                attack=attack,
                repetition=repetition,
                password=password,
                timeout_s=args.timeout,
                results_dir=str(results_dir),
                pair_prefix="main",
            )

    finally:
        print("Unloading target model:", target, flush=True)
        try:
            subprocess.run(
                ["ollama", "stop", target],
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print("GPU cleanup warning:", type(exc).__name__)

    print("T2-B MAIN EXPERIMENT BATCH FINISHED")


if __name__ == "__main__":
    main()
