"""Figure I.1. Learning gains with minimal and expanded prompts."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import matplotlib.pyplot as plt
from studentbench.plotting import (
    configure,
    save_figure,
    figure_cli,
)


def render(analysis_root, output_dir):
    configure("prompts")
    records = [
        json.loads(line)
        for line in (Path(analysis_root) / "prompts/prompt_comparison_results.jsonl")
        .read_text()
        .splitlines()
    ]
    done = {row["record_id"]: row for row in records}
    labels = {
        "claude-opus-4-8": "Opus 4.8*",
        "gemini-3.5-flash": "Gemini 3.5 Flash*",
        "gpt-5.5": "GPT-5.5*",
        "opus-5-high": "Opus 5 (high)",
        "gemini-3.6-flash-low": "Gemini 3.6 Flash (low)",
    }
    colors = {"minimal": "#0072B2", "expanded": "#D55E00"}
    fig, axes = plt.subplots(
        2, 1, figsize=(7.1, 5.5), gridspec_kw={"height_ratios": [3, 2]}, sharex=True
    )
    fig.subplots_adjust(left=0.285, right=0.895, top=0.915, bottom=0.23, hspace=0.38)
    configs = [
        (
            "pilot",
            ["claude-opus-4-8", "gemini-3.5-flash", "gpt-5.5"],
            "A  Prompt-selection pilot",
        ),
        (
            "s5",
            ["opus-5-high", "gemini-3.6-flash-low"],
            "B  Later, nonconcurrent cohorts",
        ),
    ]
    source_groups = []
    for ax, (dataset, models, title) in zip(axes, configs):
        y = np.arange(len(models))[::-1]
        ax.set_title(title, loc="left", fontweight="bold", pad=13)
        ax.set_yticks(y, labels=[labels[m] for m in models])
        ax.tick_params(axis="y", length=0, pad=10)
        ax.set_ylim(-0.6, len(models) - 0.35)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e6eaed", linewidth=0.65)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#aeb8bf")
        ax.spines["bottom"].set_linewidth(0.7)
        ax.tick_params(axis="x", color="#aeb8bf", length=3)
        ax.text(
            1.025,
            1.005,
            "Sessions",
            transform=ax.transAxes,
            fontsize=8,
            color="#4b5563",
            ha="left",
            va="bottom",
        )
        for center, model in zip(y, models):
            for public, offset, marker in [
                ("minimal", 0.14, "o"),
                ("expanded", -0.14, "D"),
            ]:
                cond = (
                    ("lean/lean" if public == "minimal" else "max/max")
                    if dataset == "pilot"
                    else public
                )
                rid = f"group:{dataset}:{model}:{cond}"
                r = done[rid]
                source_groups.append(rid)
                m = r["mean_gain_pp"]
                lo = r["ci95_low"]
                hi = r["ci95_high"]
                ax.errorbar(
                    m,
                    center + offset,
                    xerr=[[m - lo], [hi - m]],
                    fmt=marker,
                    markersize=5,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    color=colors[public],
                    ecolor=colors[public],
                    elinewidth=1.3,
                    capsize=3,
                    capthick=1,
                    label=public if center == y[0] else None,
                )
                ax.text(
                    1.025,
                    center + offset,
                    str(r["n"]),
                    transform=ax.get_yaxis_transform(),
                    fontsize=8,
                    color="#374151",
                    ha="left",
                    va="center",
                )
    axes[-1].set_xlim(-5, 31)
    axes[-1].set_xticks([0, 5, 10, 15, 20, 25, 30])
    axes[-1].set_xlabel("Learning gain (percentage points)", labelpad=8)
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0],
            [0],
            color=colors["minimal"],
            marker="o",
            markersize=5,
            linewidth=1.3,
            label="Minimal prompts",
        ),
        Line2D(
            [0],
            [0],
            color=colors["expanded"],
            marker="D",
            markersize=5,
            linewidth=1.3,
            label="Expanded prompts",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.065),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=2,
    )
    fig.text(
        0.5,
        0.039,
        "Points: cohort means; bars: 95% confidence intervals",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )
    fig.text(
        0.5,
        0.013,
        "* Pilot reasoning settings were not recorded.",
        ha="center",
        fontsize=8,
        color="#4b5563",
    )

    save_figure(fig, output_dir, "figure_24_prompt_comparisons", bbox_inches=None)


if __name__ == "__main__":
    figure_cli(render)
