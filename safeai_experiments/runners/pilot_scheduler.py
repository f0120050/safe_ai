"""Run SafeAI pilot experiments from a manifest."""

import argparse
import getpass
import json
from pathlib import Path

from safeai_experiments.runners.batch_runner import run_worker
from safeai_experiments.config.experiment_config import MODELS
from safeai_experiments.runners.ollama_model_manager import prepare_model


MANIFEST_PATH = Path(
    "safeai_experiments/config/pilot_manifest.json"
)


def load_manifest(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def load_completed(results_path):
    """Return completed experiment configurations."""
    completed = set()

    if not results_path.exists():
        return completed

    with results_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            # Do not silently discard failed or timed-out experiments.
            # Their records remain available for analysis.
            key = (
                record["pair_id"],
                record["safety_mode"],
            )
            completed.add(key)

    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["qwen", "gptoss", "granite"],
        required=True,
    )
    parser.add_argument(
        "--manifest",
        default=str(MANIFEST_PATH),
        help="Path to the experiment manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned experiments without running models.",
    )

    args = parser.parse_args()

    manifest = load_manifest(args.manifest)

    results_dir = Path(manifest["results_directory"])
    results_path = results_dir / "results.jsonl"

    experiments = [
        item for item in manifest["experiments"]
        if item["model"] == args.model
    ]

    completed = load_completed(results_path)

    pending = [
        item for item in experiments
        if (
            item["pair_id"],
            item["safety_mode"],
        ) not in completed
    ]

    print("Model:", args.model)
    print("Planned experiments:", len(experiments))
    print("Already recorded:", len(experiments) - len(pending))
    print("Pending experiments:", len(pending))

    if args.dry_run:
        for item in pending:
            print(
                item["pair_id"],
                item["safety_mode"],
                f"repetition={item['repetition']}",
            )
        return

    if not pending:
        print("Nothing to run.")
        return

    # Prepare the model once, before starting any agent executions.
    requires_agent = any(
        item["safety_mode"] in ("prompt_only", "output_only")
        for item in pending
    )

    if requires_agent:
        ollama_model = MODELS[args.model].removeprefix("ollama_chat/")

        print("\nPreparing Ollama model:", ollama_model, flush=True)

        preparation = prepare_model(
            ollama_model,
            minimum_gpu_percent=80,
        )

        print("GPU check:", preparation["reason"], flush=True)

        if not preparation["gpu_ready"]:
            raise SystemExit(
                "GPU preparation failed; no pilot runs started."
            )

    password = getpass.getpass("User Y password (once): ")

    for index, item in enumerate(pending, start=1):
        print(
            f"\n=== Pilot {index}/{len(pending)} ===",
            flush=True,
        )
        print(
            item["pair_id"],
            item["safety_mode"],
            flush=True,
        )

        # Derive the prefix from the manifest's exact pair_id.
        suffix = (
            f"-{item['attack_id']}-{item['model']}-"
            f"{item['repetition']:03d}"
        )

        if not item["pair_id"].endswith(suffix):
            raise ValueError(
                f"Invalid manifest pair_id: {item['pair_id']}"
            )

        pair_prefix = item["pair_id"][:-len(suffix)]

        if not pair_prefix:
            raise ValueError("Empty pair_id prefix.")

        run_worker(
            model=item["model"],
            mode=item["safety_mode"],
            attack=item["attack_id"],
            repetition=item["repetition"],
            password=password,
            timeout_s=manifest["timeout_seconds"],
            results_dir=str(results_dir),
            pair_prefix=pair_prefix,
        )

    print("\nPILOT SCHEDULER FINISHED")


if __name__ == "__main__":
    main()
