"""Figure 22. Conversation indicators by GRE section."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from studentbench.plotting import (
    configure,
    MODEL_TEXT,
    display_label,
    model_color,
    save_figure,
    figure_cli,
)
from matplotlib import colors
from studentbench.plotting import AXIS_GRAY, GRID_GRAY, TICK_TEXT, INTERVAL_GRAY

RULES = [
    "contingent_scaffolding",
    "self_explanation_prompts",
    "interactive_questioning",
    "productive_struggle",
    "worked_example_after_attempt",
    "metacognitive_prompting",
]
RULE_NAMES = [
    "Scaffolding\ncues",
    "Ask to\nexplain",
    "Interactive\nquestions",
    "Early\nattempt",
    "Long solution\nafter a reply",
    "Reasoning\nchecks",
]


def format_forest_axis(ax):
    """Figure 4A's typography and decoration, with categorical row axes hidden."""
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID_GRAY, linewidth=0.65)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS_GRAY)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="x", color=AXIS_GRAY, labelcolor=TICK_TEXT, width=0.7)
    ax.tick_params(axis="y", labelcolor=MODEL_TEXT)


def label(model):
    return (
        "Human tutor (spoken)"
        if model == "human"
        else (
            "Sonnet 4.6 / 5 (low)"
            if model == "sonnet-section-specific-low"
            else display_label(model)
        )
    )


def render(analysis_root, output_dir):
    configure("teaching")
    prev = pd.read_csv(
        Path(analysis_root) / "teaching/conversation_strategy_prevalence.csv"
    )
    fig = plt.figure(figsize=(7.1, 7.2))
    cmap = colors.LinearSegmentedColormap.from_list(
        "rules", ["#ffffff", GRID_GRAY, INTERVAL_GRAY, TICK_TEXT]
    )
    for ip, scope in enumerate(["quant", "verbal"]):
        b = prev[prev.scope == scope].sort_values(
            ["is_human", "mean_strategy_score"], ascending=[True, False]
        )
        n = len(b)
        base = 0.555 if ip == 0 else 0.09
        ax = fig.add_axes([0.265, base, 0.46, 0.305])
        score = fig.add_axes([0.81, base, 0.175, 0.305])
        vals = b[RULES].to_numpy() * 100
        ax.pcolormesh(
            np.arange(7) - 0.5,
            np.arange(n + 1) - 0.5,
            vals,
            cmap=cmap,
            vmin=0,
            vmax=100,
            edgecolors="white",
            lw=0.8,
        )
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_yticks(range(n), [label(m) for m in b.arm], fontsize=7.6)
        ax.set_xticks(range(6), RULE_NAMES, fontsize=6.3)
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=3)
        ax.spines[:].set_visible(False)
        for i, r in enumerate(b.itertuples()):
            for j in range(6):
                ax.text(
                    j,
                    i,
                    f"{vals[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    color="white" if vals[i, j] > 75 else TICK_TEXT,
                )
            ax.text(6.03, i, str(r.n), ha="center", va="center", fontsize=7.2)
            score.errorbar(
                100 * r.mean_strategy_score,
                i,
                xerr=[
                    [100 * (r.mean_strategy_score - r.mean_strategy_score_ci_low)],
                    [100 * (r.mean_strategy_score_ci_high - r.mean_strategy_score)],
                ],
                fmt="*" if r.is_human else "o",
                color=model_color(r.arm),
                ecolor=INTERVAL_GRAY,
                ms=6.5 if r.is_human else 4.2,
                markeredgecolor="white",
                markeredgewidth=1.05,
                capsize=2.0,
                elinewidth=1.0,
                zorder=3,
            )
        ax.text(6.03, -0.8, "n", ha="center", fontsize=7.2)
        score.set_xlim(55, 100)
        score.set_ylim(n - 0.5, -0.5)
        score.set_yticks([])
        score.set_xticks([60, 80, 100])
        score.tick_params(labelsize=7, length=2)
        format_forest_axis(score)
        score.set_xlabel("Mean (%)", fontsize=7.5)
        score.set_title("Mean prevalence\n95% interval", fontsize=7.5, pad=8)
        for tick, r in zip(ax.get_yticklabels(), b.itertuples()):
            tick.set_color("#7f2a21" if r.is_human else MODEL_TEXT)
        fig.text(
            0.035,
            base + 0.405,
            ("A   Quantitative" if ip == 0 else "B   Verbal"),
            fontsize=10,
            weight="bold",
        )
        fig.text(
            0.035,
            base + 0.375,
            "Sessions matching each conversation rule (%)",
            fontsize=8,
            color=MODEL_TEXT,
        )
    save_figure(fig, output_dir, "figure_22_conversation_by_section", bbox_inches=None)


if __name__ == "__main__":
    figure_cli(render)
