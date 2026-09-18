"""Figure 3. Lesson planning, practice design and conversation indicators."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
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
    root = Path(analysis_root) / "teaching"
    fits = {
        key: json.loads((root / f"fit_{key}.json").read_text())
        for key in ["planning_combined", "practice_combined"]
    }
    prev = pd.read_csv(root / "conversation_strategy_prevalence.csv")
    # One shared model column removes repeated names and permits direct
    # cross-panel reading while retaining each construct's original scale.
    model_text = MODEL_TEXT
    axis_gray = AXIS_GRAY
    grid_gray = GRID_GRAY
    fig = plt.figure(figsize=(10.5, 4.0))
    model_labels = []
    intervals = []
    expert_order = [r["model"] for r in fits["planning_combined"]["models"]]
    combined = prev[prev.scope == "combined"].set_index("arm")
    order = expert_order + ["human"]
    assert set(expert_order) == {
        r["model"] for r in fits["practice_combined"]["models"]
    }
    assert all(m in combined.index for m in order)
    positions = np.arange(len(order), dtype=float)
    positions[-1] += 0.45
    row_y = dict(zip(order, positions))
    # Continuous bands help readers follow the same tutor across distinct scales.
    row_background = fig.add_axes([0.004, 0.13, 0.991, 0.765], zorder=-1)
    row_background.set_ylim(positions[-1] + 0.65, -0.7)
    row_background.set_xlim(0, 1)
    row_background.axis("off")
    for y in positions[::2]:
        row_background.axhspan(
            y - 0.5, y + 0.5, color="#91afc3", alpha=0.1, linewidth=0
        )
    axes = []

    def format_axis(ax):
        ax.set_facecolor("none")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=grid_gray, linewidth=0.65)
        ax.set_yticks([])
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.spines["bottom"].set_color(axis_gray)
        ax.spines["bottom"].set_linewidth(0.7)
        ax.tick_params(
            axis="x",
            labelsize=12,
            color=axis_gray,
            labelcolor="#38444c",
            length=3.5,
            width=0.8,
        )
        ax.set_ylim(positions[-1] + 0.65, -0.7)

    def mark(ax, panel, x, lo, hi, y, model):
        ax.errorbar(
            x,
            y,
            xerr=[[x - lo], [hi - x]],
            fmt="*" if model == "human" else "o",
            color=model_color(model),
            ecolor=INTERVAL_GRAY,
            ms=9 if model == "human" else 6.1,
            markeredgecolor="white",
            markeredgewidth=1.05,
            elinewidth=1.0,
            capsize=2.3,
            capthick=1.0,
            zorder=4,
        )
        intervals.append(
            {
                "panel": panel,
                "model": model,
                "x": float(x),
                "lo": float(lo),
                "hi": float(hi),
                "y": float(y),
            }
        )

    for left, key, title, letter in [
        (0.185, "planning_combined", "Lesson planning", "A"),
        (0.455, "practice_combined", "Practice design", "B"),
    ]:
        ax = fig.add_axes([left, 0.13, 0.255, 0.765])
        axes.append(ax)
        ft = fits[key]
        rs = ft["models"]
        format_axis(ax)
        ax.set_xlim(-1.8, 1.85)
        ax.set_xticks([-1, 0, 1])
        assert all(-1.8 < r["lo"] <= r["ability"] <= r["hi"] < 1.85 for r in rs)
        for r in rs:
            y = row_y[r["model"]]
            mark(ax, letter, r["ability"], r["lo"], r["hi"], y, r["model"])
        ax.set_title(
            f"{letter}   {title}", loc="left", fontsize=14, weight="bold", pad=10
        )
    names_ax = fig.add_axes([0.025, 0.13, 0.145, 0.765])
    names_ax.set_ylim(axes[0].get_ylim())
    names_ax.set_xlim(0, 1)
    names_ax.axis("off")
    for m in order:
        annotation = names_ax.text(
            0.08,
            row_y[m],
            label(m),
            ha="left",
            va="center",
            fontsize=9,
            color="#7f2a21" if m == "human" else model_text,
            clip_on=False,
            gid="model_label_teaching_" + m,
        )
        model_labels.append(annotation)
    ax = fig.add_axes([0.725, 0.13, 0.270, 0.765])
    axes.append(ax)
    format_axis(ax)
    ax.set_xlim(56.5, 92)
    ax.set_xticks([60, 70, 80, 90])
    for m in order:
        r = combined.loc[m]
        x = 100 * r.mean_strategy_score
        lo = 100 * r.mean_strategy_score_ci_low
        hi = 100 * r.mean_strategy_score_ci_high
        assert 56.5 < lo <= x <= hi < 92
        mark(ax, "C", x, lo, hi, row_y[m], m)
    ax.set_title(
        "C   Conversational pedagogy", loc="left", fontsize=14, weight="bold", pad=10
    )
    ax.set_xlabel(
        "Conversation indicators (%)", fontsize=11.5, color="black", labelpad=7
    )
    shared_xlabel = fig.text(
        0.4475,
        0.036,
        "Bradley–Terry pairwise expert preference score (log-odds)",
        ha="center",
        fontsize=11.5,
        color="black",
    )
    # A single rule separates the spoken human reference; missing expert scores
    # remain blank rather than being placed at zero.
    for target in [names_ax] + axes:
        target.axhline(positions[-1] - 0.8, color=axis_gray, linewidth=0.65, zorder=1)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_position = ax.xaxis.label.get_transform().transform(
        ax.xaxis.label.get_position()
    )
    shared_xlabel.set_y(fig.transFigure.inverted().transform(label_position)[1])
    shared_xlabel.set_va(ax.xaxis.label.get_va())
    boxes = [t.get_window_extent(renderer) for t in model_labels]
    overlapping = []
    outside = []
    for i, box in enumerate(boxes):
        if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(
            box.x1, box.y1
        ):
            outside.append(model_labels[i].get_text())
        assert box.x1 < axes[0].bbox.x0 - 4, model_labels[i].get_text()
        for j, other in enumerate(boxes[:i]):
            if box.overlaps(other):
                overlapping.append(
                    [model_labels[j].get_text(), model_labels[i].get_text()]
                )
    assert not overlapping, overlapping
    assert not outside, outside
    save_figure(
        fig,
        output_dir,
        "figure_03_teaching_evaluations",
        pad_inches=0.04,
        extra_canvas=(1.284141, 0.727968),
    )


if __name__ == "__main__":
    figure_cli(render)
