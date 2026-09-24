"""Figure F.3. Student-disputed practice answers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
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
    rates = pd.read_csv(Path(analysis_root) / "teaching/flag_rates.csv")
    """Figure 3's shared-row layout, retaining the existing dispute estimates."""
    fig = plt.figure(figsize=(10.5, 3.65))
    # Combined-rate order is unchanged; each scope retains its own x scale.
    order = (
        rates.query("scope=='combined'")
        .sort_values("flags_per_100_answered")
        .model_preset.tolist()
    )
    positions = np.arange(len(order), dtype=float)
    ylimits = (positions[-1] + 0.65, -0.7)
    row_background = fig.add_axes([0.004, 0.13, 0.991, 0.765], zorder=-1)
    row_background.set_ylim(ylimits)
    row_background.set_xlim(0, 1)
    row_background.axis("off")
    for y in positions[::2]:
        row_background.axhspan(
            y - 0.5, y + 0.5, color="#91afc3", alpha=0.1, linewidth=0
        )
    axes = []
    intervals = []
    model_labels = []
    limits = []
    for j, s in enumerate(["quant", "verbal", "combined"]):
        b = rates[rates.scope == s].copy()
        b["display_id"] = b.model_preset.replace(
            {
                "sonnet-4.6-low": "sonnet-section-specific-low",
                "sonnet-5-low": "sonnet-section-specific-low",
            }
        )
        b = b.set_index("display_id").loc[order]
        assert len(b) == len(order) and b.index.is_unique
        ax = fig.add_axes(
            [[0.185, 0.455, 0.725][j], 0.13, 0.255 if j < 2 else 0.270, 0.765]
        )
        axes.append(ax)
        ax.set_facecolor("none")
        format_forest_axis(ax)
        for i, r in enumerate(b.itertuples()):
            x = r.flags_per_100_answered
            ax.errorbar(
                x,
                i,
                xerr=[[x - r.ci_low_per_100], [r.ci_high_per_100 - x]],
                fmt="o",
                ms=6.1,
                color=model_color(r.model_preset),
                ecolor=INTERVAL_GRAY,
                markeredgecolor="white",
                markeredgewidth=1.05,
                capsize=2.3,
                capthick=1.0,
                elinewidth=1.0,
                zorder=4,
            )
            intervals.append(
                {
                    "panel": chr(65 + j),
                    "scope": s,
                    "model": r.model_preset,
                    "display_id": r.Index,
                    "x": float(x),
                    "lo": float(r.ci_low_per_100),
                    "hi": float(r.ci_high_per_100),
                    "y": float(i),
                    "n_students": int(r.n_students),
                    "n_answered": int(r.n_answered),
                    "n_flagged": int(r.n_flagged),
                }
            )
        limit = (
            math.ceil(float(b.ci_high_per_100.max()) / 2) * 2
            if s != "verbal"
            else math.ceil(float(b.ci_high_per_100.max()) * 2) / 2
        )
        ax.set_xlim(-0.025 * limit, limit * 1.04)
        ax.set_xticks([0, limit / 2, limit])
        ax.set_ylim(ylimits)
        ax.set_yticks([])
        ax.tick_params(
            axis="x",
            labelsize=12,
            color=AXIS_GRAY,
            labelcolor=TICK_TEXT,
            length=3.5,
            width=0.8,
        )
        limits.append(list(ax.get_xlim()))
        names = {"quant": "Quantitative", "verbal": "Verbal", "combined": "Combined"}
        ax.set_title(
            chr(65 + j) + "   " + names[s],
            loc="left",
            fontsize=14,
            weight="bold",
            pad=10,
        )
    names_ax = fig.add_axes([0.025, 0.13, 0.145, 0.765])
    names_ax.set_ylim(ylimits)
    names_ax.set_xlim(0, 1)
    names_ax.axis("off")
    for y, m in zip(positions, order):
        annotation = names_ax.text(
            0.08,
            y,
            label(m).removesuffix("*"),
            ha="left",
            va="center",
            fontsize=9,
            color=MODEL_TEXT,
            clip_on=False,
            gid="model_label_disputes_" + m,
        )
        model_labels.append(annotation)
    # Counts, resampling details and the mixed Sonnet identity belong in the caption.
    fig.text(
        0.59,
        0.016,
        "Student-disputed answers per 100 answered problems",
        ha="center",
        fontsize=11.5,
        color="black",
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in model_labels]
    for i, box in enumerate(boxes):
        assert box.x1 < axes[0].bbox.x0 - 4, model_labels[i].get_text()
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(
            box.x1, box.y1
        ), model_labels[i].get_text()
        assert not any(box.overlaps(other) for other in boxes[:i]), model_labels[
            i
        ].get_text()
    save_figure(fig, output_dir, "figure_23_disputed_answers", pad_inches=0.04)


if __name__ == "__main__":
    figure_cli(render)
