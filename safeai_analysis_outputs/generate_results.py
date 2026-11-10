import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("safeai_experiments/main_results")
OUT = Path("safeai_analysis_outputs")
OUT.mkdir(exist_ok=True)

MODELS = {
    "qwen": "Qwen 2.5 7B",
    "gptoss": "GPT-OSS 20B",
    "granite": "Granite 8B",
}

MODES = {
    "prompt_only": "Prompt only",
    "input_only": "Input Security",
    "output_only": "Output Security",
    "input_output": "Input + Output",
    "execution_only": "Execution Security",
}

SOURCES = {
    "T2": {
        "prompt_only": "T2",
        "input_only": "T2",
        "output_only": "T2",
        "input_output": "T2",
        "execution_only": "T2_execution_rerun/T2",
    },
    "T2B": {
        "prompt_only": "T2B",
        "input_only": "T2B",
        "output_only": "T2B",
        "input_output": "T2B",
        "execution_only": "T2B_execution/T2B",
    },
    "T4": {
        "prompt_only": "T4",
        "input_only": "T4",
        "execution_only": "T4",
    },
}

def load_group(scenario, model, mode):
    base = ROOT / (
        "T4_rerun"
        if scenario == "T4" and model == "qwen"
        else SOURCES[scenario][mode]
    )
    path = base / model / "results.jsonl"

    if not path.is_file():
        raise FileNotFoundError(path)

    selected = {}
    technical_errors = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            if r.get("attack_id") != scenario:
                continue
            if r.get("safety_mode") != mode:
                continue

            if r.get("status") == "TECHNICAL_ERROR":
                technical_errors += 1
                continue

            if r.get("timeout_occurred") is True or r.get("error_type"):
                raise ValueError(
                    f"Unresolved error in {path}: {r.get('run_id')}"
                )

            repetition = r.get("repetition")

            if repetition is None:
                raise ValueError(f"Missing repetition in {path}")

            if repetition in selected:
                raise ValueError(
                    f"Duplicate valid repetition {repetition} in {path}"
                )

            selected[repetition] = r

    rows = list(selected.values())

    if len(rows) != 30:
        raise ValueError(
            f"{scenario}/{model}/{mode}: "
            f"expected 30 valid repetitions, found {len(rows)} "
            f"(excluded technical errors: {technical_errors})"
        )

    for r in rows:
        if not isinstance(r.get("unauthorized_http_access"), bool):
            raise ValueError(f"Invalid HTTP access field: {path}")
        if not isinstance(r.get("total_time_s"), (int, float)):
            raise ValueError(f"Invalid execution time: {path}")

    return rows, technical_errors


def percentage(rows, field):
    values = [r.get(field) for r in rows]
    if not all(isinstance(x, bool) for x in values):
        return None
    return 100 * sum(values) / len(values)


def summarize():
    records = []

    for scenario, modes in SOURCES.items():
        for model in MODELS:
            for mode in modes:
                rows, excluded = load_group(scenario, model, mode)

                times = [r["total_time_s"] for r in rows]

                record = {
                    "scenario": scenario,
                    "model": MODELS[model],
                    "mode": MODES[mode],
                    "n": len(rows),
                    "unauthorized_http_pct": percentage(
                        rows, "unauthorized_http_access"
                    ),
                    "final_disclosure_pct": (
                        percentage(rows, "unauthorized_final_disclosure")
                        if scenario != "T4" else None
                    ),
                    "state_change_pct": (
                        percentage(rows, "task_completed")
                        if scenario == "T4" else None
                    ),
                    "blocked_pct": (
                        100 * sum(r.get("status") == "BLOCKED"
                                  for r in rows) / len(rows)
                    ),
                    "mean_time_s": statistics.mean(times),
                    "sd_time_s": statistics.stdev(times),
                    "excluded_technical_errors": excluded,
                }

                records.append(record)

    return records


def number(x):
    return "--" if x is None else f"{x:.1f}"


def latex_table(scenario, records):
    items = [r for r in records if r["scenario"] == scenario]

    if scenario == "T4":
        columns = (
            "Model & Safety mode & $n$ & HTTP (\\%) & "
            "State change (\\%) & Blocked (\\%) & Time (s)"
        )
        alignment = "llrrrrl"
    else:
        columns = (
            "Model & Safety mode & $n$ & HTTP (\\%) & "
            "Disclosure (\\%) & Blocked (\\%) & Time (s)"
        )
        alignment = "llrrrrl"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Security effectiveness and execution time for {scenario}.}}",
        rf"\label{{tab:{scenario.lower()}_results}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        columns + r" \\",
        r"\hline",
    ]

    for r in items:
        outcome = (
            r["state_change_pct"]
            if scenario == "T4"
            else r["final_disclosure_pct"]
        )

        time_text = (
            f'{r["mean_time_s"]:.2f} '
            + r'$\pm$ '
            + f'{r["sd_time_s"]:.2f}'
        )

        cells = [
            r["model"],
            r["mode"],
            str(r["n"]),
            number(r["unauthorized_http_pct"]),
            number(outcome),
            number(r["blocked_pct"]),
            time_text,
        ]

        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    (OUT / f"table_{scenario}.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )


def plot_metric(scenario, records, metric, filename, ylabel):
    items = [r for r in records if r["scenario"] == scenario]
    modes = list(SOURCES[scenario])

    x = np.arange(len(modes))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10, 4.7))

    for index, model in enumerate(MODELS.values()):
        model_rows = {
            r["mode"]: r for r in items if r["model"] == model
        }

        values = [
            model_rows[MODES[m]][metric]
            for m in modes
        ]

        errors = (
            [
                model_rows[MODES[m]]["sd_time_s"]
                for m in modes
            ]
            if metric == "mean_time_s" else None
        )

        positions = x + (index - 1) * width

        ax.bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=3 if errors is not None else 0,
            label=model,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODES[m] for m in modes],
        rotation=18,
        ha="right"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(f"{scenario}: {ylabel}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    if metric == "unauthorized_http_pct":
        ax.set_ylim(0, 105)

    fig.tight_layout()

    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(
        OUT / f"{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def main():
    records = summarize()

    csv_path = OUT / "table_all_results.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    for scenario in SOURCES:
        latex_table(scenario, records)

        plot_metric(
            scenario, records,
            "unauthorized_http_pct",
            f"security_{scenario}",
            "Unauthorized HTTP activity (%)"
        )

        plot_metric(
            scenario, records,
            "mean_time_s",
            f"latency_{scenario}",
            "Mean execution time (s)"
        )

    print("Analysis completed.")
    print("Valid experiments:", sum(r["n"] for r in records))
    print("Groups:", len(records))
    print("Output directory:", OUT)
    print("LaTeX tables: 3")
    print("PDF figures: 6")
    print("PNG figures: 6")
    print("CSV summary: 1")


if __name__ == "__main__":
    main()
