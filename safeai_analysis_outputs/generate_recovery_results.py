"""Generate publication-ready T4-R tables and figures."""

import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(
    "safeai_experiments/recovery_results/T4_R_main"
)

OUT = Path("safeai_analysis_outputs")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "qwen": "Qwen 2.5 7B",
    "gptoss": "GPT-OSS 20B",
    "granite": "Granite 8B",
}

EXPECTED_REPETITIONS = set(range(1, 31))


def load_results(model_key):
    path = ROOT / model_key / "results.jsonl"

    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as f:
        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    repetitions = [r.get("repetition") for r in rows]

    if len(rows) != 30:
        raise ValueError(
            f"{model_key}: Expected 30 records, "
            f"found {len(rows)}"
        )

    if set(repetitions) != EXPECTED_REPETITIONS:
        raise ValueError(
            f"{model_key}: Missing or duplicate repetitions"
        )

    required_fields = (
        "state",
        "status",
        "state_before_attack",
        "state_after_attack",
        "state_after_recovery",
        "unauthorized_put_200",
        "rollback_success",
        "notification_created",
        "total_time_s",
    )

    for r in rows:
        missing = [k for k in required_fields if k not in r]
        if missing:
            raise ValueError(
                f"{model_key}: Missing fields {missing}"
            )

        if r.get("schema_version") != "T4-R-v1":
            raise ValueError(
                f"{model_key}: Unexpected schema version"
            )

        if r.get("safety_mode") != "recovery_only":
            raise ValueError(
                f"{model_key}: Unexpected safety mode"
            )

        if r.get("status") == "TECHNICAL_ERROR":
            raise ValueError(
                f"{model_key}: Technical error present"
            )

        t = r["total_time_s"]
        if isinstance(t, bool) or not isinstance(t, (int, float)):
            raise ValueError(
                f"{model_key}: Invalid execution time"
            )

    return rows


def percent(count, total):
    return 100.0 * count / total


def summarize(model_key, rows):
    n = len(rows)

    unauthorized_put = sum(
        r["unauthorized_put_200"] is True
        for r in rows
    )

    verified_transitions = sum(
        r["state_before_attack"] == 1
        and r["state_after_attack"] == 2
        and r["state_after_recovery"] == 1
        for r in rows
    )

    rollback_success = sum(
        r["rollback_success"] is True
        for r in rows
    )

    notifications = sum(
        r["notification_created"] is True
        for r in rows
    )

    rolled_back = sum(
        r["state"] == "ROLLED_BACK"
        for r in rows
    )

    blocked = sum(
        r["status"] == "BLOCKED"
        for r in rows
    )

    times = [float(r["total_time_s"]) for r in rows]

    return {
        "model": MODELS[model_key],
        "n": n,
        "unauthorized_put_count": unauthorized_put,
        "unauthorized_put_pct": percent(unauthorized_put, n),
        "verified_transition_count": verified_transitions,
        "verified_transition_pct": percent(verified_transitions, n),
        "rollback_count": rollback_success,
        "rollback_pct": percent(rollback_success, n),
        "notification_count": notifications,
        "notification_pct": percent(notifications, n),
        "rolled_back_count": rolled_back,
        "blocked_count": blocked,
        "mean_time_s": statistics.mean(times),
        "sd_time_s": statistics.stdev(times),
    }


def write_csv(results):
    path = OUT / "table_T4_R.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    return path


def write_latex(results):
    path = OUT / "table_T4_R.tex"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{T4-R post-execution recovery and "
            r"incident notification results across three LLMs. "
            r"Notifications are simulated.}"
        ),
        r"\label{tab:t4_recovery_results}",
        r"\begin{tabular}{lrrrrrl}",
        r"\hline",
        (
            r"Model & $n$ & PUT (\%) & "
            r"Verified restoration (\%) & "
            r"Rollback (\%) & Notification (\%) & Time (s) \\"
        ),
        r"\hline",
    ]

    for r in results:
        time_text = (
            f'{r["mean_time_s"]:.2f}'
            + r" $\pm$ "
            + f'{r["sd_time_s"]:.2f}'
        )

        cells = [
            r["model"],
            str(r["n"]),
            f'{r["unauthorized_put_pct"]:.1f}',
            f'{r["verified_transition_pct"]:.1f}',
            f'{r["rollback_pct"]:.1f}',
            f'{r["notification_pct"]:.1f}',
            time_text,
        ]

        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return path


def save_figure(fig, filename):
    fig.tight_layout()

    fig.savefig(
        OUT / f"{filename}.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        OUT / f"{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_latency(results):
    labels = [r["model"] for r in results]
    means = [r["mean_time_s"] for r in results]
    errors = [r["sd_time_s"] for r in results]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    x = np.arange(len(labels))

    ax.bar(
        x,
        means,
        yerr=errors,
        capsize=5,
        width=0.55
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("End-to-end execution time (s)")
    ax.set_title("T4-R: Recovery experiment execution time")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    save_figure(fig, "recovery_latency_T4_R")


def plot_success(results):
    labels = [r["model"] for r in results]

    metrics = [
        ("Verified restoration", "verified_transition_pct"),
        ("Rollback success", "rollback_pct"),
        ("Simulated notification", "notification_pct"),
    ]

    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for index, (label, key) in enumerate(metrics):
        values = [r[key] for r in results]
        positions = x + (index - 1) * width

        ax.bar(
            positions,
            values,
            width=width,
            label=label
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("T4-R: Recovery and notification outcomes")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    save_figure(fig, "recovery_success_T4_R")


def main():
    results = []

    print("T4-R RECOVERY ANALYSIS")
    print("=" * 60)

    for model_key in MODELS:
        rows = load_results(model_key)
        summary = summarize(model_key, rows)
        results.append(summary)

        print(f"\n{summary['model']}")
        print("Valid experiments:", summary["n"])
        print("Unauthorized PUT:", summary["unauthorized_put_count"])
        print("Verified 1 -> 2 -> 1:", summary["verified_transition_count"])
        print("Rollback success:", summary["rollback_count"])
        print("Notification created:", summary["notification_count"])
        print(
            "Execution time:",
            f"{summary['mean_time_s']:.2f} "
            f"+/- {summary['sd_time_s']:.2f} s"
        )

    # Generate outputs only after validating every model.
    write_csv(results)
    write_latex(results)
    plot_latency(results)
    plot_success(results)

    print("\n" + "=" * 60)
    print("Analysis completed successfully.")
    print("Total valid experiments:", sum(r["n"] for r in results))
    print("Output folder:", OUT)
    print("Generated: 1 LaTeX table, 1 CSV, 2 PDF and 2 PNG figures")


if __name__ == "__main__":
    main()
