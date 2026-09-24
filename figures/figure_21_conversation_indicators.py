"""Figure F.1. Combined conversation indicators and uncertainty."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from studentbench.plotting import (
    configure,
    HUMAN_RED,
    display_label,
    model_color,
    save_figure,
    figure_cli,
)
from matplotlib import colors, ticker

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


def render(analysis_root, output_dir):
    configure("main")
    b = (
        pd.read_csv(
            Path(analysis_root) / "teaching/conversation_strategy_prevalence.csv"
        )
        .query("scope == 'combined'")
        .sort_values(["is_human", "mean_strategy_score"], ascending=[True, False])
    )
    # Keep the figure's height in the full-width paper after shorter human labels
    # reduce its tight-cropped width; preserve every data row and font size.
    n = len(b)
    fig = plt.figure(figsize=(11.45, 7.15))
    ax = fig.add_axes([0.23, 0.12, 0.505, 0.68])
    score = fig.add_axes([0.795, 0.12, 0.20, 0.68])
    vals = b[RULES].to_numpy() * 100
    cmap = colors.LinearSegmentedColormap.from_list(
        "native_gray", ["#ffffff", "#e6eaed", "#86939d", "#38444c"]
    )
    ax.pcolormesh(
        np.arange(7) - 0.5,
        np.arange(n + 1) - 0.5,
        vals,
        cmap=cmap,
        vmin=0,
        vmax=100,
        edgecolors="white",
        linewidth=1.8,
    )
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(n - 0.5, -0.5)
    names = [
        (
            (
                "Sonnet 4.6 / 5 (low)"
                if x == "sonnet-section-specific-low"
                else display_label(x)
            )
            if not h
            else "Human tutor (spoken)"
        )
        for x, h in zip(b.arm, b.is_human)
    ]
    ax.set_yticks(range(n), names, fontsize=14)
    ax.set_xticks(
        range(6),
        [
            "Scaffolding\ncues",
            "Ask to\nexplain",
            "Interactive\nquestions",
            "Early\nattempt\nrequest",
            "Long\nsolution\nafter reply",
            "Reasoning\nchecks",
        ],
        fontsize=13.2,
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=7)
    ax.spines[:].set_visible(False)
    for edge in [0.5, 1.5, 2.5, 3.5, 4.5]:
        ax.plot([edge, edge], [-0.54, -2.35], color="#e6eaed", lw=0.65, clip_on=False)
    for i in range(n):
        for j in range(6):
            ax.text(
                j,
                i,
                f"{vals[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=14,
                color="white" if vals[i, j] > 75 else "#38444c",
            )
        ax.text(
            6.02, i, str(int(b.iloc[i]["n"])), ha="center", va="center", fontsize=13.2
        )
        r = b.iloc[i]
        mean = 100 * r.mean_strategy_score
        lo = 100 * r.mean_strategy_score_ci_low
        hi = 100 * r.mean_strategy_score_ci_high
        color = HUMAN_RED if r.is_human else model_color(r.arm)
        score.errorbar(
            mean,
            i,
            xerr=[[mean - lo], [hi - mean]],
            fmt="*" if r.is_human else "o",
            ms=8.2 if r.is_human else 5.3,
            mfc=color,
            mec="white",
            mew=1.05,
            ecolor="#86939d",
            capsize=2.3,
            elinewidth=1.0,
            zorder=3,
        )
    # The common rule-prevalence mean is bounded by 0--100%; this displayed
    # interval zoom (55--95) makes its actual sampling precision legible.
    score.set_xlim(55, 95)
    score.set_ylim(n - 0.5, -0.5)
    score.set_yticks([])
    score.set_xticks([60, 70, 80, 90])
    score.xaxis.set_major_formatter(ticker.PercentFormatter(100, decimals=0))
    score.tick_params(labelsize=13.2)
    score.set_axisbelow(True)
    score.grid(axis="x", color="#e6eaed", lw=0.65)
    score.spines[["top", "right", "left"]].set_visible(False)
    score.spines["bottom"].set_color("#aeb8bf")
    score.spines["bottom"].set_linewidth(0.7)
    score.tick_params(axis="x", color="#aeb8bf", labelcolor="#38444c", width=0.7)
    score.set_title("Mean prevalence\n95% interval", fontsize=14, pad=10, color="black")
    for tick, human in zip(ax.get_yticklabels(), b.is_human):
        tick.set_color("#7f2a21" if human else "#4b5964")
    ax.tick_params(axis="x", labelcolor="#38444c")
    ax.text(6.02, -0.94, "n", ha="center", fontsize=13.2)
    separator_y = ax.get_position().y0 + ax.get_position().height / n
    fig.canvas.draw()
    label_left = min(
        t.get_window_extent(fig.canvas.get_renderer()).x0 for t in ax.get_yticklabels()
    )
    separator_left = fig.transFigure.inverted().transform((label_left, 0))[0]
    fig.add_artist(
        plt.Line2D(
            [separator_left, 0.995],
            [separator_y, separator_y],
            transform=fig.transFigure,
            color="#aeb8bf",
            lw=0.7,
        )
    )
    title = "Automatically detected conversation patterns"
    fig.text(0.025, 0.965, title, weight="bold", fontsize=16, va="top", color="black")
    fig.text(
        0.025,
        0.914,
        "Percentage of sessions matching each of six fixed language and turn-order rules",
        fontsize=14,
        color="#4b5964",
    )
    fig.text(
        0.025,
        0.035,
        "Automatic patterns are not validated teaching quality; AI chat and spoken human tutoring differ in format.",
        fontsize=12.5,
        color="#4b5964",
    )
    save_figure(fig, output_dir, "figure_21_conversation_indicators")


if __name__ == "__main__":
    figure_cli(render)
