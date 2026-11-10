"""Run the main SafeAI experiments, one model per batch."""

import argparse
import getpass
import json
import subprocess
from pathlib import Path

from safeai_experiments.config.experiment_config import (
    MODELS,
    SAFETY_MODES,
)
from safeai_experiments.runners.batch_runner import run_worker
from safeai_experiments.runners.ollama_model_manager import (
    gpu_ready,
    prepare_model,
    stop_all_models,
)


RESULTS_ROOT = Path("safeai_experiments/main_results")


def recorded_pairs(path):
    """Return recorded experiment configurations, including failures."""
    keys = set()

    if not path.exists():
        return keys

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            keys.add((record["pair_id"], record["safety_mode"]))

    return keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--attack", default="T2")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=RESULTS_ROOT,
        help="Root directory for experiment results.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.repeats < 1 or args.timeout < 1:
        parser.error("--repeats and --timeout must be positive")

    results_dir = args.results_root / args.attack / args.model
    results_path = results_dir / "results.jsonl"
    completed = recorded_pairs(results_path)

    # Balanced, reproducible mode ordering.
    # Each mode occupies every position once per four repetitions.
    plan = []
    modes = list(SAFETY_MODES)

    for repetition in range(1, args.repeats + 1):
        offset = (repetition - 1) % len(modes)
        ordered_modes = modes[offset:] + modes[:offset]

        for mode in ordered_modes:
            plan.append((repetition, mode))

    pending = [
        (repetition, mode)
        for repetition, mode in plan
        if (f"main-{args.attack}-{args.model}-{repetition:03d}", mode)
        not in completed
    ]

    print("Model:", args.model)
    print("Attack:", args.attack)
    print("Planned:", len(plan))
    print("Recorded:", len(plan) - len(pending))
    print("Pending:", len(pending))
    print("Results:", results_dir)

    if args.dry_run or not pending:
        return

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

        password = getpass.getpass("User Y password (once): ")

        for index, (repetition, mode) in enumerate(pending, 1):
            ready, reason = gpu_ready(target)

            if not ready:
                raise RuntimeError(
                    f"GPU residency lost before run {index}: {reason}"
                )

            print(
                f"\n=== {index}/{len(pending)} "
                f"{args.model} {mode} repetition={repetition} ===",
                flush=True,
            )

            run_worker(
                model=args.model,
                mode=mode,
                attack=args.attack,
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
            print("GPU cleanup warning:", exc, flush=True)

    print("MAIN EXPERIMENT BATCH FINISHED")


if __name__ == "__main__":
    main()
