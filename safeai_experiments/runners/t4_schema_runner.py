"""Select the original T4 or T4-R recovery experiment."""

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


CONFIGS = {
    "original": {
        "worker": "safeai_experiments.runners.t4_worker",
        "modes": ("prompt_only", "input_only", "execution_only"),
        "root": "safeai_experiments/main_results/T4",
    },
    "recovery": {
        "worker": "safeai_experiments.runners.t4_recovery_worker",
        "modes": ("recovery_only",),
        "root": "safeai_experiments/recovery_results/T4_R",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema", required=True, choices=CONFIGS
    )
    parser.add_argument(
        "--model", required=True,
        choices=("qwen", "gptoss", "granite")
    )
    parser.add_argument("--mode", required=True)
    parser.add_argument(
        "--repetition", type=int, default=1
    )
    args = parser.parse_args()

    config = CONFIGS[args.schema]

    if args.mode not in config["modes"]:
        parser.error(
            f"Mode {args.mode!r} is not valid "
            f"for schema {args.schema!r}"
        )

    if args.repetition < 1:
        parser.error("Repetition must be positive")

    if importlib.util.find_spec(config["worker"]) is None:
        parser.error(
            f"Worker not yet implemented: {config['worker']}"
        )

    results_dir = (
        Path(config["root"]) / args.model
    )

    print("Schema:", args.schema)
    print("Worker:", config["worker"])
    print("Results directory:", results_dir)

    env = os.environ.copy()
    env["SAFEAI_RESULTS_DIR"] = str(results_dir)

    command = [
        sys.executable,
        "-m", config["worker"],
        "--model", args.model,
        "--mode", args.mode,
        "--repetition", str(args.repetition),
    ]

    subprocess.run(command, env=env, check=True)


if __name__ == "__main__":
    main()
