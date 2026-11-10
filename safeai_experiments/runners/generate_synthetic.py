from pathlib import Path
import shutil

from safeai_experiments.config.result_schema import ExperimentResult
from safeai_experiments.runners.result_writer import ResultWriter


RESULTS_DIR = Path("safeai_experiments/synthetic_results")
MODELS = ["qwen2.5:7b", "gpt-oss:20b", "granite4.2:8b"]
MODES = ["prompt_only", "input_only", "output_only"]

if RESULTS_DIR.exists() and any(RESULTS_DIR.iterdir()):
    raise SystemExit("Synthetic results directory is not empty.")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
writer = ResultWriter(results_dir=RESULTS_DIR)

count = 0

for model_index, model in enumerate(MODELS):
    for repetition in range(1, 4):
        pair_id = f"T2-{model_index}-{repetition}"

        for mode in MODES:
            run_id = f"synthetic-{pair_id}-{mode}"

            # Simulated model behavior; not actual test results.
            attack_succeeded = repetition != 3
            technical_failure = (
                model_index == 2
                and repetition == 3
                and mode == "output_only"
            )

            if mode == "input_only":
                status = "BLOCKED"
                executed = False
                unauthorized_access = False
                disclosure = False
                duration = 0.12
            elif technical_failure:
                status = "TIMEOUT"
                executed = True
                unauthorized_access = None
                disclosure = None
                duration = 600.0
            else:
                executed = True
                unauthorized_access = attack_succeeded
                disclosure = attack_succeeded and mode == "prompt_only"
                status = (
                    "BLOCKED"
                    if attack_succeeded and mode == "output_only"
                    else "COMPLETED"
                )
                duration = (
                    [12.0, 26.0, 42.0][model_index]
                    + repetition * 1.5
                )

            result = ExperimentResult(
                run_id=run_id,
                pair_id=pair_id,
                model=model,
                attack_id="T2",
                safety_mode=mode,
                repetition=repetition,
                status=status,
                input_decision="DENY" if mode == "input_only" else None,
                execution_decision=(
                    "DENY" if mode == "output_only" and attack_succeeded
                    and not technical_failure else None
                ),
                output_decision=(
                    "DENY" if status == "BLOCKED" and mode == "output_only"
                    else None
                ),
                agent_executed=executed,
                unauthorized_http_access=unauthorized_access,
                unauthorized_final_disclosure=disclosure,
                total_time_s=duration,
                input_check_time_s=0.12 if mode == "input_only" else None,
                output_check_time_s=(
                    0.18 if mode == "output_only" and not technical_failure
                    else None
                ),
                json_error_count=1 if technical_failure else 0,
                timeout_occurred=technical_failure,
                agent_step_count=0 if not executed else 2,
                task_completed=status == "COMPLETED",
                error_type="Timeout" if technical_failure else None,
            )

            writer.save(result)
            count += 1

print(f"Generated {count} SYNTHETIC experiment records.")
print("CSV:", writer.csv_path)
print("JSONL:", writer.jsonl_path)
