"""Figure F.1. Inference cost per point of learning gain."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from studentbench.plotting import (
    configure,
    COLORS,
    HUMAN_RED,
    display_label,
    save_figure,
    figure_cli,
)
from matplotlib import ticker
from matplotlib.patches import ConnectionPatch, Rectangle


def render(analysis_root, output_dir):
    """Draw complete paired-bootstrap intervals from fresh session cost estimates."""
    configure("cost")
    table = pd.read_csv(Path(analysis_root) / "costs/cost_efficiency_by_arm_scope.csv")
    selected = table[table.scope.isin(["quant", "verbal"])]

    def label(row):
        return (
            "Human tutor" if row["kind"] == "human" else display_label(row["arm_label"])
        )

    def color(row):
        return HUMAN_RED if row["kind"] == "human" else COLORS[row["family"]]

    fig = plt.figure(figsize=(10.5, 9.8))
    for scope, title, bottom, axis_end, axis_step, zoom_end, zoom_step in [
        ("quant", "A   Quantitative", 0.565, 6, 1, 0.50, 0.10),
        ("verbal", "B   Verbal", 0.085, 8, 2, 0.75, 0.15),
    ]:
        rows = (
            selected[selected.scope == scope]
            .sort_values("cost_per_gain_pp", ascending=False)
            .to_dict("records")
        )
        assert len(rows) == 14 and rows[0]["kind"] == "human"
        # Preserve Figure 5's named reference in each section; this does not
        # claim that it establishes equivalence within either section.
        baseline_id = "gemini-3.5-flash-low"
        baseline_row = next(r for r in rows if r["arm_id"] == baseline_id)
        baseline_cost = baseline_row["cost_per_gain_pp"]
        zoom_rows = [
            (i, r) for i, r in enumerate(rows) if r["cost_per_gain_pp"] < zoom_end
        ]
        assert len(zoom_rows) == 12
        assert max(r["cost_per_gain_pp_ci_high"] for r in rows) < axis_end
        assert all(r["cost_per_gain_pp_ci_high"] < zoom_end for _, r in zoom_rows)
        ax = fig.add_axes([0.16, bottom, 0.73, 0.35])
        ax.set_xlim(0, axis_end)
        ax.set_ylim(len(rows) - 0.35, -0.65)
        ax.set_yticks(range(len(rows)), labels=[])
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e6eaed", linewidth=0.65)
        ax.grid(axis="y", color="#e6eaed", linewidth=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#aeb8bf")
            ax.spines[side].set_linewidth(0.7)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(axis_step))
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:.2f}"))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.tick_params(
            axis="x", width=0.8, color="#aeb8bf", length=3.5, pad=3.5, labelsize=12
        )
        ax.set_xlabel(
            "Cost per percentage point of learning gain (USD)",
            fontsize=11.5,
            labelpad=9,
        )
        ax.set_title(title, loc="left", fontsize=14, weight="bold", pad=12)
        price_column_x = (0.938 - 0.16) / 0.73
        multiplier_column_x = (0.997 - 0.16) / 0.73
        gain_column_x = (1.060 - 0.16) / 0.73
        ax.annotate(
            "USD/point",
            (price_column_x, 1),
            xycoords="axes fraction",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=9,
            ha="right",
            va="bottom",
            color="#4b5964",
            annotation_clip=False,
        )
        ax.annotate(
            "Cost\nmultiple",
            (multiplier_column_x, 1),
            xycoords="axes fraction",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=9,
            linespacing=1.4,
            ha="right",
            va="bottom",
            color="#4b5964",
            annotation_clip=False,
        )
        ax.annotate(
            "Learning\ngain (pp)",
            (gain_column_x, 1),
            xycoords="axes fraction",
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=9,
            linespacing=1.4,
            ha="right",
            va="bottom",
            color="#4b5964",
            annotation_clip=False,
        )

        names, prices, plotted, multipliers, gains = [], [], [], [], []

        def display_cost(value):
            return f"${value:.3f}" if value < 0.10 else f"${value:.2f}"

        def display_multiplier(value):
            # Keep large comparisons memorable and distinguish nearby costs around 1x.
            number = (
                f"{value:.0f}"
                if value >= 10
                else f"{value:.2f}".rstrip("0").rstrip(".")
            )
            return number + "×"

        def row_y(index):
            return index

        def plot_row(target, index, row, panel):
            y = row_y(index)
            mean = row["cost_per_gain_pp"]
            lo, hi = row["cost_per_gain_pp_ci_low"], row["cost_per_gain_pp_ci_high"]
            target.errorbar(
                mean,
                y,
                xerr=[[mean - lo], [hi - mean]],
                fmt="*" if row["kind"] == "human" else "o",
                color=color(row),
                markersize=np.sqrt(155 if row["kind"] == "human" else 48),
                markeredgecolor="white",
                markeredgewidth=1.05,
                elinewidth=1.0,
                capsize=2.4,
                capthick=1.0,
                clip_on=False,
                zorder=4,
            )
            if panel == "main":
                text = target.annotate(
                    label(row),
                    (0, y),
                    xytext=(-6, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=9,
                    color="#7f2a21" if row["kind"] == "human" else "#4b5964",
                    zorder=6,
                    annotation_clip=False,
                )
                names.append(text)
            plotted.append(
                {
                    "display_row": index + 1,
                    "cost_rank_cheapest_first": len(rows) - index,
                    "scope": scope,
                    "arm_id": row["arm_id"],
                    "label": label(row),
                    "y": y,
                    "x": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "display_cost": display_cost(mean),
                    "color": color(row),
                    "panel": panel,
                }
            )

        for index, row in enumerate(rows):
            plot_row(ax, index, row, "main")
            prices.append(
                ax.text(
                    price_column_x,
                    row_y(index),
                    display_cost(row["cost_per_gain_pp"]),
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=9,
                    color="#4b5964",
                    clip_on=False,
                )
            )
            multiplier = row["cost_per_gain_pp"] / baseline_cost
            multiplier_text = display_multiplier(multiplier)
            ax.text(
                multiplier_column_x,
                row_y(index),
                multiplier_text,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                color="#4b5964",
                clip_on=False,
                weight="bold" if row["arm_id"] == baseline_id else "normal",
            )
            multipliers.append(
                {
                    "arm_id": row["arm_id"],
                    "unrounded_multiplier": multiplier,
                    "display_multiplier": multiplier_text,
                    "baseline": row["arm_id"] == baseline_id,
                }
            )
            gain_text = f"{row['mean_gain_pp']:.1f}"
            ax.text(
                gain_column_x,
                row_y(index),
                gain_text,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=9,
                color="#4b5964",
                clip_on=False,
            )
            gains.append(
                {
                    "arm_id": row["arm_id"],
                    "mean_gain_pp": row["mean_gain_pp"],
                    "display_gain": gain_text,
                }
            )

        # Preserve rank order in a compact zoom, leaving the expensive
        # references unobscured. Its rows are compressed, not vertically aligned.
        zoom_top, zoom_bottom = 1.45, len(rows) - 0.45
        inset_bounds = [0.25, 0.16, 0.42, 0.49]
        inset = ax.inset_axes(inset_bounds)
        inset.set_zorder(10)
        inset.set_facecolor("white")
        inset.set_xlim(0, zoom_end)
        inset.set_ylim(zoom_bottom, zoom_top)
        inset.set_yticks([i for i, _ in zoom_rows], labels=[])
        inset.tick_params(axis="y", length=0)
        inset.set_axisbelow(True)
        inset.grid(axis="x", color="#e6eaed", linewidth=0.65)
        inset.grid(axis="y", color="#e6eaed", linewidth=0.65)
        inset.xaxis.set_major_locator(ticker.MultipleLocator(zoom_step))
        inset.xaxis.set_major_formatter(ticker.StrMethodFormatter("${x:.2f}"))
        inset.xaxis.set_minor_locator(ticker.NullLocator())
        inset.tick_params(
            axis="x", color="#aeb8bf", width=0.8, length=3.5, labelsize=8, pad=3.5
        )
        inset.text(
            0.02,
            0.98,
            "Zoom",
            transform=inset.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="black",
            weight="normal",
            zorder=12,
        )
        for spine in inset.spines.values():
            spine.set_color("#7F8C95")
            spine.set_linewidth(0.75)
        for index, row in zoom_rows:
            plot_row(inset, index, row, "zoom")

        # Figure 4A's solid gray outlines and visibly separated round dots.
        ax.add_patch(
            Rectangle(
                (0, zoom_top),
                zoom_end,
                zoom_bottom - zoom_top,
                fill=False,
                edgecolor="#7F8C95",
                linewidth=0.75,
                clip_on=False,
                zorder=1.4,
                gid="cost_zoom_source_outline_" + scope,
            )
        )
        for side, y, inset_y_coord in [
            ("top", zoom_top, 1),
            ("bottom", zoom_bottom, 0),
        ]:
            ax.add_artist(
                ConnectionPatch(
                    xyA=(zoom_end, y),
                    coordsA=ax.transData,
                    xyB=(0, inset_y_coord),
                    coordsB=inset.transAxes,
                    color="#7F8C95",
                    linewidth=0.9,
                    linestyle=(0, (0.1, 4.0)),
                    capstyle="round",
                    clip_on=False,
                    zorder=1.4,
                    gid=f"cost_zoom_connector_{scope}_{side}",
                )
            )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inset_box = inset.get_window_extent(renderer)
        assert all(
            name.get_window_extent(renderer).x1 < ax.bbox.x0 - 2 for name in names
        ), "A model name meets the main plotting area"
        for index, row in enumerate(rows):
            lo, hi = row["cost_per_gain_pp_ci_low"], row["cost_per_gain_pp_ci_high"]
            endpoints = ax.transData.transform([(lo, index), (hi, index)])
            assert not (
                inset_box.y0 <= endpoints[0, 1] <= inset_box.y1
                and endpoints[0, 0] <= inset_box.x1
                and endpoints[1, 0] >= inset_box.x0
            ), "The inset covers a main estimate or interval"

    save_figure(fig, output_dir, "figure_16_section_cost_per_gain", pad_inches=0.04)


if __name__ == "__main__":
    figure_cli(render)
