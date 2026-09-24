"""Figure D.3. Student engagement versus AI reply time."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from studentbench.plotting import (
    configure,
    COLORS,
    MODEL_TEXT,
    PARETO_BLUE,
    display_label,
    save_figure,
    figure_cli,
)
from matplotlib import ticker
from studentbench.plotting import (
    AXIS_GRAY,
    GRID_GRAY,
    SECTION_NAMES,
)
import matplotlib

SCOPES = ["quant", "verbal", "combined"]
TITLES = SECTION_NAMES


def axis_style(ax, grid="both"):
    """Figure 4A's shared frame, tick and grid treatment."""
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(AXIS_GRAY)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(color=AXIS_GRAY, labelcolor="#38444c", width=0.7)
    ax.grid(axis=grid, color=GRID_GRAY, lw=0.65)


def place_labels(ax, rows, xkey, ykey, fontsize=9):
    # Adjacent, fully named labels; no connectors or data displacement.
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    scale = ax.figure.dpi / 72
    points = [
        matplotlib.transforms.Bbox.from_bounds(px - 4, py - 4, 8, 8)
        for px, py in ax.transData.transform([[r[xkey], r[ykey]] for r in rows])
    ]
    occupied = []
    metadata = []
    for r in sorted(rows, key=lambda z: -z[ykey]):
        x, y = r[xkey], r[ykey]
        px, py = ax.transData.transform((x, y))
        t = ax.text(
            x,
            y,
            display_label(r["arm_label"]),
            fontsize=fontsize,
            color=MODEL_TEXT,
            zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.87},
        )
        choices = []
        horizontal_pad = (
            10
            if r.get("scope") == "quant" and r["arm_id"] == "gemini-3.1-pro-high"
            else 5
        )
        for d in [5, 10, 16, 23, 31, 39, 47]:
            for dx, dy, ha, va in [
                (1, 1, "left", "bottom"),
                (-1, 1, "right", "bottom"),
                (1, -1, "left", "top"),
                (-1, -1, "right", "top"),
                (0, 1, "center", "bottom"),
                (0, -1, "center", "top"),
            ]:
                pos = ax.transData.inverted().transform(
                    (px + dx * horizontal_pad * scale, py + dy * d * scale)
                )
                t.set_position(pos)
                t.set_ha(ha)
                t.set_va(va)
                bb = t.get_window_extent(renderer).expanded(1.01, 1.08)
                outside = sum(
                    [
                        max(0, ax.bbox.x0 + 1 - bb.x0),
                        max(0, bb.x1 - ax.bbox.x1 + 1),
                        max(0, ax.bbox.y0 + 1 - bb.y0),
                        max(0, bb.y1 - ax.bbox.y1 + 1),
                    ]
                )
                hits = sum(bb.overlaps(b) for b in occupied)
                ph = sum(bb.overlaps(b) for b in points)
                choices.append(
                    (
                        outside * 1e6 + hits * 1e5 + ph * 1e4 + d,
                        pos,
                        ha,
                        va,
                        bb,
                        hits,
                        ph,
                        outside,
                    )
                )
        _, pos, ha, va, bb, hits, ph, out = min(choices, key=lambda z: z[0])
        t.set_position(pos)
        t.set_ha(ha)
        t.set_va(va)
        occupied.append(bb)
        metadata.append(
            {
                "arm_id": r["arm_id"],
                "label": t.get_text(),
                "x": x,
                "y": y,
                "label_overlaps": hits,
                "marker_overlaps": ph,
                "outside_px": out,
            }
        )
    return metadata


def render(analysis_root, output_dir):
    configure("efficiency")
    root = Path(analysis_root) / "costs"
    ENGAGE = pd.read_csv(root / "latency_student_engagement_by_arm_scope.csv")
    STAT = json.loads((root / "statistics.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.8))
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.16, top=0.74, wspace=0.21)
    meta = []
    for ax, scope in zip(axes, SCOPES):
        b = ENGAGE[ENGAGE.scope == scope]
        x = b.median_latency_s.to_numpy()
        y = b.mean_student_messages.to_numpy()
        a = STAT["latency_student_engagement"]["associations"][scope]
        ax.set_xscale("log")
        ax.set_xlim(min(x) / 1.9, max(x) * 1.8)
        ax.set_ylim(min(y) - 4.5, max(y) + 4.5)
        xx = np.geomspace(min(x), max(x), 80)
        co = np.polyfit(np.log10(x), y, 1)
        ax.plot(xx, np.polyval(co, np.log10(xx)), lw=1.8, color=PARETO_BLUE, zorder=1)
        ax.scatter(
            x,
            y,
            s=40,
            color=[COLORS[r["family"]] for r in b.to_dict("records")],
            edgecolor="white",
            linewidth=1.05,
            zorder=3,
        )
        axis_style(ax)
        ax.tick_params(labelsize=13)
        ax.set_xticks([2, 5, 10, 20, 40], ["2", "5", "10", "20", "40"])
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        ax.set_xlabel("AI reply time (seconds; log scale)", fontsize=13)
        if scope == "quant":
            ax.set_ylabel("Student messages per session", fontsize=14)
        ax.set_title(
            TITLES[scope] + f"  (k={len(b)})",
            loc="left",
            fontsize=13,
            weight="bold",
            pad=54,
        )
        ax.text(
            0,
            1.025,
            f"Pearson r={a['pearson_r']:.2f}, p={a['pearson_p_holm_6']:.4f}\nSpearman ρ={a['spearman_rho']:.2f}, p={a['spearman_p_holm_6']:.4f}".replace(
                "=0.", "=."
            ),
            transform=ax.transAxes,
            fontsize=10.2,
            color=MODEL_TEXT,
            linespacing=1.5,
        )
        meta.append(
            {
                "scope": scope,
                "tests": a,
                "points": place_labels(
                    ax,
                    b.to_dict("records"),
                    "median_latency_s",
                    "mean_student_messages",
                    11,
                ),
            }
        )
    fig.suptitle(
        "Student engagement vs. AI reply time",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=15,
        weight="bold",
    )
    save_figure(fig, output_dir, "figure_18_reply_time_engagement")


if __name__ == "__main__":
    figure_cli(render)
