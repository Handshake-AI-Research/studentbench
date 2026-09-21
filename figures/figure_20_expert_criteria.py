"""Figure G.1. Eight expert criteria and three overall rankings."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import matplotlib.pyplot as plt
from studentbench.plotting import (
    configure,
    display_label,
    model_color,
    save_figure,
    figure_cli,
)

KEYS = [
    "relevant_concepts",
    "concept_grouping",
    "concept_prioritization",
    "time_allocation",
    "practice_alignment",
    "difficulty_appropriateness",
    "answer_key_accuracy",
    "test_taking_strategies",
    "common8_quant",
    "common8_verbal",
    "common8_combined",
]
TITLES = [
    "Relevant\nconcepts",
    "Concept\ngrouping",
    "Concept\nprioritization",
    "Time\nallocation",
    "Practice\nalignment",
    "Appropriate\ndifficulty",
    "Example\naccuracy",
    "Test-taking\nstrategies",
    "Quantitative overall",
    "Verbal overall",
    "Combined overall",
]
WIDTH = 756.0
HEIGHT = 792.0
PRINT_WIDTH = 482.4
SCALE = PRINT_WIDTH / WIDTH
PRINT_FONTS = {
    "model": 6.75081982,
    "tick": 7.66775976,
    "title": 8.94572,
    "axis": 8.39036,
}
NATIVE_FONTS = {k: v / SCALE for k, v in PRINT_FONTS.items()}


def render(analysis_root, output_dir):
    configure("criteria")
    root = Path(analysis_root) / "teaching"
    fits = {key: json.loads((root / f"fit_{key}.json").read_text()) for key in KEYS}
    order = [row["model"] for row in fits["common8_combined"]["models"]]
    names = {model: display_label(model) for model in order}
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72))
    positions = np.arange(13, dtype=float)
    ylim = (12.65, -0.7)
    # Three label columns, one per row, keep all eleven plots aligned to the same
    # pooled ordering without repeating names in each individual panel.
    row_tops = [55.0, 318.0, 561.0]
    plot_height = 190.0
    panel_rows = [range(4), range(4, 8), range(8, 11)]
    label_artists = []
    data_records = []
    for row, top in enumerate(row_tops):
        base = (HEIGHT - top - plot_height) / HEIGHT
        height = plot_height / HEIGHT
        bg = fig.add_axes([0.004, base, 0.991, height], zorder=-1)
        bg.set_ylim(ylim)
        bg.set_xlim(0, 1)
        bg.axis("off")
        for y in positions[::2]:
            bg.axhspan(y - 0.5, y + 0.5, color="#91afc3", alpha=0.1, linewidth=0)
        label_ax = fig.add_axes([0, base, 0.17, height])
        label_ax.set_ylim(ylim)
        label_ax.set_xlim(0, 1)
        label_ax.axis("off")
        for y, model in zip(positions, order):
            label_artists.append(
                label_ax.text(
                    22 / (WIDTH * 0.17),
                    y,
                    names[model],
                    ha="left",
                    va="center",
                    fontsize=NATIVE_FONTS["model"],
                    color="#4b5964",
                    gid=f"model_443_row{row}_{model}",
                    clip_on=False,
                )
            )
        lefts, widths = (
            ([0.185, 0.39, 0.595, 0.80], [0.195] * 4)
            if row < 2
            else ([0.185, 0.455, 0.725], [0.255, 0.255, 0.270])
        )
        for col, panel in enumerate(panel_rows[row]):
            key = KEYS[panel]
            letter = chr(65 + panel)
            ax = fig.add_axes([lefts[col], base, widths[col], height])
            ax.set_facecolor("none")
            ax.set_ylim(ylim)
            ax.set_yticks([])
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="#e6eaed", linewidth=0.65)
            ax.spines[["left", "right", "top"]].set_visible(False)
            ax.spines["bottom"].set_color("#aeb8bf")
            ax.spines["bottom"].set_linewidth(0.7)
            bounds = (-2.5, 2.5)
            ticks = [-2, 0, 2]
            ax.set_xlim(bounds)
            ax.set_xticks(ticks)
            ax.tick_params(
                axis="x",
                labelsize=NATIVE_FONTS["tick"],
                length=3.5,
                width=0.8,
                color="#aeb8bf",
                labelcolor="#38444c",
            )
            bymodel = {r["model"]: r for r in fits[key]["models"]}
            assert set(bymodel) <= set(order)
            for y, model in zip(positions, order):
                if model not in bymodel:
                    continue  # Preserve blank model cells from the original section fits.
                r = bymodel[model]
                x, lo, hi = r["ability"], r["lo"], r["hi"]
                assert bounds[0] < lo <= x <= hi < bounds[1]
                ax.errorbar(
                    x,
                    y,
                    xerr=[[x - lo], [hi - x]],
                    fmt="o",
                    ms=6.1,
                    color=model_color(model),
                    ecolor="#86939d",
                    markeredgecolor="white",
                    markeredgewidth=1.05,
                    elinewidth=1,
                    capsize=2.3,
                    capthick=1,
                    zorder=4,
                )
                record = {
                    "panel": letter,
                    "fit": key,
                    "model": model,
                    "x": x,
                    "lo": lo,
                    "hi": hi,
                    "y": float(y),
                }
                data_records.append(record)
            # Separate panel letter/title preserves the original hanging indent.
            title_lines = TITLES[panel].split("\n")
            line_step = NATIVE_FONTS["title"] * 1.35
            first_baseline = top - 14 - (len(title_lines) - 1) * line_step
            fig.text(
                lefts[col],
                1 - first_baseline / HEIGHT,
                letter,
                ha="left",
                va="baseline",
                fontsize=NATIVE_FONTS["title"],
                weight="bold",
                gid="panel_letter_" + letter,
            )
            for line_i, line in enumerate(title_lines):
                fig.text(
                    lefts[col] + 20 / WIDTH,
                    1 - (first_baseline + line_i * line_step) / HEIGHT,
                    line,
                    ha="left",
                    va="baseline",
                    fontsize=NATIVE_FONTS["title"],
                    weight="bold",
                    gid=f"panel_title_{letter}_{line_i}",
                )
    fig.text(
        0.59,
        7 / HEIGHT,
        "Bradley–Terry pairwise expert preference score (log-odds)",
        ha="center",
        va="baseline",
        fontsize=NATIVE_FONTS["axis"],
        color="black",
        gid="shared_bt_axis_title",
    )
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(ren) for t in label_artists]
    assert all(b.x1 < WIDTH * fig.dpi / 72 * 0.185 - 2 for b in boxes), (
        "Model label reaches first plot"
    )
    for i, b in enumerate(boxes):
        assert not any(b.overlaps(a) for a in boxes[:i]), "Model labels overlap"

    save_figure(fig, output_dir, "figure_20_expert_criteria", bbox_inches=None)


if __name__ == "__main__":
    figure_cli(render)
