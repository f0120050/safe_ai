import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest
from scipy.stats import fisher_exact
from math import sqrt


BASE = Path("safeai_experiments")
TECHNICAL_FAILURES = {"TIMEOUT", "MODEL_ERROR", "TOOL_ERROR", "TECHNICAL_ERROR"}


def read_results(path):
    df = pd.read_csv(path)

    # Never mix synthetic schema checks with real experiments.
    df = df[
        ~df["run_id"].str.startswith("schema-test-", na=False)
    ].copy()

    if df.empty:
        raise ValueError("No experiment records found.")

    for field in [
        "unauthorized_http_access",
        "unauthorized_final_disclosure",
    ]:
        df[field] = df[field].map(
            lambda x: (
                True if str(x).lower() == "true"
                else False if str(x).lower() == "false"
                else pd.NA
            )
        )

    df["technical_failure"] = df["status"].isin(
        TECHNICAL_FAILURES
    )

    return df



def wilson_interval(successes, total, z=1.959963984540054):
    """95% Wilson score confidence interval for a binary proportion."""
    if total == 0:
        return float("nan"), float("nan")

    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    margin = (
        z * sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        ) / denominator
    )
    return 100 * (center - margin), 100 * (center + margin)


def calculate_summary(df):
    rows = []

    for (model, mode), group in df.groupby(
        ["model", "safety_mode"]
    ):
        valid = group[~group["technical_failure"]]

        http = valid["unauthorized_http_access"].dropna()
        disclosure = valid[
            "unauthorized_final_disclosure"
        ].dropna()

        durations = pd.to_numeric(
            valid["total_time_s"], errors="coerce"
        ).dropna()

        rows.append({
            "model": model,
            "safety_mode": mode,
            "runs": len(group),
            "technical_failures": int(
                group["technical_failure"].sum()
            ),
            "http_observed": len(http),
            "http_violations": int(http.sum()),
            "http_ci_lower": wilson_interval(
                int(http.sum()), len(http)
            )[0],
            "http_ci_upper": wilson_interval(
                int(http.sum()), len(http)
            )[1],
            "http_violation_rate": (
                100 * http.mean() if len(http) else float("nan")
            ),
            "disclosure_observed": len(disclosure),
            "disclosure_count": int(disclosure.sum()),
            "disclosure_ci_lower": wilson_interval(
                int(disclosure.sum()), len(disclosure)
            )[0],
            "disclosure_ci_upper": wilson_interval(
                int(disclosure.sum()), len(disclosure)
            )[1],
            "disclosure_rate": (
                100 * disclosure.mean()
                if len(disclosure) else float("nan")
            ),
            "mean_time_s": (
                durations.mean()
                if len(durations) else float("nan")
            ),
            "median_time_s": (
                durations.median()
                if len(durations) else float("nan")
            ),
        })

    return pd.DataFrame(rows)


def plot_metric(summary, column, ylabel, filename, output_dir):
    chart = summary.pivot(
        index="model",
        columns="safety_mode",
        values=column,
    )

    ax = chart.plot(kind="bar", figsize=(10, 5))
    ax.set_title(ylabel + " — SYNTHETIC DATA")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Safety mode")
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=160)
    plt.close()



def plot_rate_with_ci(summary, prefix, ylabel, filename, output_dir):
    import numpy as np

    models = list(summary["model"].unique())
    modes = list(summary["safety_mode"].unique())
    x = np.arange(len(models))
    width = 0.8 / len(modes)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, mode in enumerate(modes):
        subset = summary.set_index(["model", "safety_mode"])
        positions = x - 0.4 + width * (i + 0.5)

        for j, model in enumerate(models):
            row = subset.loc[(model, mode)]
            total = int(row[f"{prefix}_observed"])
            count_key = (
                "http_violations" if prefix == "http"
                else "disclosure_count"
            )
            count = int(row[count_key])

            if total == 0:
                continue

            rate_key = (
                "http_violation_rate" if prefix == "http"
                else "disclosure_rate"
            )
            rate = float(row[rate_key])
            lower = max(0.0, float(row[f"{prefix}_ci_lower"]))
            upper = min(100.0, float(row[f"{prefix}_ci_upper"]))

            color = f"C{i}"
            ax.bar(positions[j], rate, width=width * 0.9,
                   color=color)

            ax.errorbar(
                positions[j], rate,
                yerr=[[max(0, rate - lower)],
                      [max(0, upper - rate)]],
                fmt="none", ecolor="black",
                capsize=4, linewidth=1.3
            )

            ax.annotate(
                f"{count}/{total}",
                (positions[j], max(rate, upper)),
                xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=9
            )

        ax.bar([float("nan")], [0], color=f"C{i}", label=mode)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 115)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_title(ylabel + " — 95% Wilson CI — SYNTHETIC DATA")
    ax.legend(title="Safety mode")
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)


def exact_mcnemar(df, first, second, outcome):
    """
    Exploratory exact McNemar comparison.

    Requires unique pair_id values per condition.
    pair_id alone does not establish genuine experimental pairing.
    """
    relevant = df[
        df["safety_mode"].isin([first, second])
        & ~df["technical_failure"]
    ].copy()

    results = []

    for model, group in relevant.groupby("model"):
        if group.duplicated(
            ["pair_id", "safety_mode"]
        ).any():
            continue

        wide = group.pivot(
            index="pair_id",
            columns="safety_mode",
            values=outcome,
        )

        if first not in wide or second not in wide:
            continue

        wide = wide[[first, second]].dropna()

        b = int(((wide[first] == True)
                 & (wide[second] == False)).sum())

        c = int(((wide[first] == False)
                 & (wide[second] == True)).sum())

        discordant = b + c

        p_value = (
            min(1.0, 2 * binomtest(
                min(b, c), discordant, 0.5,
                alternative="less"
            ).pvalue)
            if discordant else 1.0
        )

        results.append({
            "model": model,
            "comparison": f"{first} vs {second}",
            "matched_pairs": len(wide),
            "b": b,
            "c": c,
            "p_value": p_value,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["synthetic", "real"],
        default="synthetic",
    )

    args = parser.parse_args()

    if args.dataset == "synthetic":
        source = BASE / "synthetic_results/results.csv"
    else:
        source = BASE / "results/results.csv"

    output_dir = BASE / "evaluation" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    df = read_results(source)
    summary = calculate_summary(df)

    summary.to_csv(
        output_dir / "summary.csv", index=False
    )

    print("\nEXPERIMENT SUMMARY")
    print(summary.to_string(index=False))

    plot_rate_with_ci(
        summary,
        "http",
        "Unauthorized HTTP Access (%)",
        "http_violations.png",
        output_dir,
    )

    plot_rate_with_ci(
        summary,
        "disclosure",
        "Unauthorized Final Disclosure (%)",
        "final_disclosures.png",
        output_dir,
    )

    plot_metric(
        summary,
        "median_time_s",
        "Median Execution Time (s) — Excluding Technical Failures",
        "execution_time.png",
        output_dir,
    )

    plot_metric(
        summary,
        "technical_failures",
        "Technical Failure Count",
        "technical_failures.png",
        output_dir,
    )

    # McNemar is only valid when runs are genuinely paired.
    # For synthetic records, this verifies the analysis pipeline.
    if args.dataset == "synthetic":
        comparisons = exact_mcnemar(
            df,
            "prompt_only",
            "output_only",
            "unauthorized_final_disclosure",
        )

        comparisons.to_csv(
            output_dir / "mcnemar_demo.csv",
            index=False,
        )

        print("\nEXACT McNEMAR — SYNTHETIC DEMONSTRATION")
        print(comparisons.to_string(index=False))

    print("\nOutput directory:", output_dir)
    print("Analysis completed.")


if __name__ == "__main__":
    main()
