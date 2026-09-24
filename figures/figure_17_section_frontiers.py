"""Figure D.2. Cost and reply time learning frontiers by GRE section."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from studentbench.plotting import (
    configure,
    PARETO_BLUE,
    PARETO_TEXT,
    PARETO_SLATE,
    HUMAN_RED,
    AXIS_GRAY,
    GRID_GRAY,
    display_label,
    save_frontier_figure,
    figure_cli,
)

PANEL_TITLE_SIZE = 14


def pareto_panel(ax, rows, metric, title):
    key = {
        "cost": "mean_cost_usd",
        "latency": "median_latency_s",
        "engagement": "mean_student_messages",
    }[metric]
    flag = "frontier_" + metric
    ai = [r for r in rows if r["kind"] == "ai"]
    frontier = sorted([r for r in ai if r[flag]], key=lambda r: r[key])
    xs = np.array([float(r[key]) for r in rows])
    ys = np.array([float(r["mean_gain_pp"]) for r in rows])
    assert len(rows) and np.isfinite(xs).all() and np.isfinite(ys).all()
    if metric in ["cost", "latency"]:
        assert (xs > 0).all()
        ax.set_xscale("log")
        lx = np.log10(xs)
        pad = max(float(np.ptp(lx)) * 0.11, 0.09)
        ax.set_xlim(10 ** (float(lx.min()) - pad), 10 ** (float(lx.max()) + pad))
        ticks = (
            [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
            if metric == "cost"
            else [1, 2, 5, 10, 20, 40, 60]
        )
        # Decades make the several-orders-of-magnitude cost scale easy to scan.
        if metric == "cost":
            ticks = [0.1, 1, 10, 100]
        lo, hi = ax.get_xlim()
        ax.xaxis.set_major_locator(
            ticker.FixedLocator([x for x in ticks if lo <= x <= hi])
        )
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(
                (lambda v, _: f"${v:g}")
                if metric == "cost"
                else (lambda v, _: f"{v:g}")
            )
        )
        ax.xaxis.set_minor_locator(ticker.NullLocator())
    else:
        span = max(float(np.ptp(xs)), 1.0)
        ax.set_xlim(
            max(0, float(xs.min()) - 0.12 * span), float(xs.max()) + 0.12 * span
        )
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, integer=True))
    span = max(float(np.ptp(ys)), 2.5)
    pad = max(span * 0.30, 1.0)
    # Retain lower padding; reduce visible upper headroom by about 40%.
    ax.set_ylim(float(ys.min()) - pad, float(ys.max()) + 0.55 * pad)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(color="#e6eaed", lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#aeb8bf")
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(axis="both", labelsize=12, color="#aeb8bf", labelcolor="#38444c")
    ax.set_title(title, loc="left", weight="bold", fontsize=15, pad=12, color="black")
    ax.set_ylabel(
        "Observed gain (percentage points)", fontsize=13, labelpad=9, color="black"
    )
    ax.set_xlabel(
        {
            "cost": "Estimated session cost (USD; logarithmic scale)",
            "latency": "Full reply time (seconds; logarithmic scale)",
            "engagement": "Student messages per session",
        }[metric],
        fontsize=13,
        labelpad=9,
        color="black",
    )
    # Extend the decorative wash to the axes boundary, then fade it to white
    # in SVG. This avoids a false vertical boundary at the final frontier point.
    # The wash is not an uncertainty interval or an additional frontier segment.
    ax.fill_between(
        [ax.get_xlim()[0], *[r[key] for r in frontier], ax.get_xlim()[1]],
        [
            frontier[0]["mean_gain_pp"],
            *[r["mean_gain_pp"] for r in frontier],
            frontier[-1]["mean_gain_pp"],
        ],
        ax.get_ylim()[0],
        color=PARETO_BLUE,
        lw=0,
        zorder=0,
        gid=f"pareto_wash_{rows[0]['scope']}_{metric}",
    )
    ax.plot(
        [r[key] for r in frontier],
        [r["mean_gain_pp"] for r in frontier],
        color=PARETO_BLUE,
        lw=2.2,
        ls=(0, (5, 3)),
        zorder=2,
    )
    texts = []
    metadata = []
    for i, r in enumerate(rows):
        human = r["kind"] == "human"
        on_frontier = bool(r[flag]) and not human
        x = float(r[key])
        y = float(r["mean_gain_pp"])
        c = HUMAN_RED if human else PARETO_BLUE if on_frontier else PARETO_SLATE
        ax.scatter(
            [x],
            [y],
            s=155 if human else 83 if on_frontier else 48,
            marker="*" if human else "o",
            color=c,
            edgecolor="white",
            linewidth=1.05,
            zorder=5 if on_frontier or human else 3,
        )
        label = "Human tutor" if human else display_label(r["arm_label"])
        badge = (
            {
                "boxstyle": "round,pad=.20",
                "facecolor": "white",
                "edgecolor": "#e4e9ed",
                "linewidth": 0.45,
                "alpha": 0.97,
            }
            if on_frontier
            else None
        )
        text = ("\u00a0" * 4 + label) if on_frontier else label
        t = ax.text(
            x,
            y,
            text,
            fontsize=11.9 if on_frontier else 11.5,
            fontweight="normal",
            color="#202d36" if on_frontier else "#7f2a21" if human else PARETO_TEXT,
            ha="left",
            va="bottom",
            zorder=6,
            bbox=badge,
        )
        texts.append(t)
        metadata.append(
            {
                "arm_id": r["arm_id"],
                "name": label,
                "scope": r["scope"],
                "metric": metric,
                "x": x,
                "gain_pp": y,
                "frontier": on_frontier,
                "human_reference": human,
                "n_students": r["n_students"],
                "family": r["family"],
            }
        )
    # Labels stay directly beside their own points. Discrete nearby placements
    # are tested against every marker and already placed label; no leaders.
    renderer = ax.figure.canvas.get_renderer()
    pixel_per_pt = ax.figure.dpi / 72
    point_boxes = [
        matplotlib.transforms.Bbox.from_bounds(px - 6, py - 6, 12, 12)
        for px, py in ax.transData.transform(np.c_[xs, ys])
    ]
    occupied = []
    order = sorted(
        range(len(rows)),
        key=lambda i: (
            not metadata[i]["frontier"],
            not metadata[i]["human_reference"],
            not (
                metadata[i]["scope"] == "verbal"
                and metric == "cost"
                and metadata[i]["arm_id"] == "gemini-3.6-flash-low"
            ),
            -metadata[i]["gain_pp"],
        ),
    )
    for i in order:
        t = texts[i]
        r = metadata[i]
        px, py = ax.transData.transform((r["x"], r["gain_pp"]))
        choices = []
        alignments = [
            (1, 1, "left", "bottom"),
            (-1, 1, "right", "bottom"),
            (1, -1, "left", "top"),
            (-1, -1, "right", "top"),
            (0, 1, "center", "bottom"),
            (0, -1, "center", "top"),
        ]
        if r["human_reference"]:
            alignments = [
                (-1, -1, "right", "top"),
                (-1, 1, "right", "bottom"),
            ] + alignments
        distances = [7, 11, 17, 24, 32, 40]
        if r["scope"] == "combined" and metric == "latency":
            if r["arm_id"] == "gpt-5.4-mini-none":
                alignments = [(1, -0.5, "left", "center")]
                distances = [3, 4, 5, 7]
            elif r["arm_id"] in ["gemini-3.6-flash-low", "opus-5-high"]:
                alignments = [(1, -1, "left", "top")]
        if r["scope"] == "quant" and metric == "latency":
            # These adjacent points need labels on opposite sides without leaders.
            if r["arm_id"] == "gpt-5.4-mini-none":
                alignments = [(1, 1, "left", "bottom")]
                distances = [3, 4, 5, 7]
            elif r["arm_id"] == "gemini-3.6-flash-low":
                alignments = [(1, -1, "left", "top")]
        if r["scope"] == "verbal" and metric in ["cost", "latency"]:
            # Keep these neighboring models attached to their own markers.
            if r["arm_id"] == "gpt-5.4-mini-none":
                alignments = [(1, 0, "left", "center")]
                distances = [3]
            elif r["arm_id"] == "gemini-3.7-flash-medium":
                alignments = [(1, 1, "left", "bottom")]
                distances = [3, 4, 5, 7]
            elif r["arm_id"] == "gemini-3.6-flash-low":
                alignments = [(1, -1, "left", "top")]
        for distance in distances:
            for dx, dy, ha, va in alignments:
                pos = ax.transData.inverted().transform(
                    (px + dx * 7 * pixel_per_pt, py + dy * distance * pixel_per_pt)
                )
                t.set_position(pos)
                t.set_ha(ha)
                t.set_va(va)
                box = t.get_window_extent(renderer).expanded(
                    1.025, 1.24 if r["frontier"] else 1.10
                )
                outside = sum(
                    [
                        max(0, ax.bbox.x0 + 2 - box.x0),
                        max(0, box.x1 - (ax.bbox.x1 - 2)),
                        max(0, ax.bbox.y0 + 2 - box.y0),
                        max(0, box.y1 - (ax.bbox.y1 - 2)),
                    ]
                )
                label_hits = sum(box.overlaps(b) for b in occupied)
                point_hits = sum(box.overlaps(b) for b in point_boxes)
                score = (
                    outside * 1e6
                    + label_hits * 1e5
                    + point_hits * 1e4
                    + distance
                    + len(choices) * 0.01
                )
                choices.append(
                    (score, pos, ha, va, box, label_hits, point_hits, outside)
                )
        _, pos, ha, va, box, label_hits, point_hits, outside = min(
            choices, key=lambda x: x[0]
        )
        t.set_position(pos)
        t.set_ha(ha)
        t.set_va(va)
        occupied.append(box)
        r["label_xy"] = list(map(float, pos))
        r["label_overlaps"] = int(label_hits)
        r["marker_overlaps"] = int(point_hits)
        r["label_outside_px"] = float(outside)
        if r["frontier"]:
            box = t.get_window_extent(renderer)
            size = box.height * 0.86
            a = ax.transAxes.inverted().transform(
                (box.x0 + 2, box.y0 + (box.height - size) / 2)
            )
            b = ax.transAxes.inverted().transform(
                (box.x0 + 2 + size, box.y0 + (box.height + size) / 2)
            )
            gid = f"pareto_icon_{r['scope']}_{metric}_{i}"
            ax.add_patch(
                plt.Rectangle(
                    a,
                    b[0] - a[0],
                    b[1] - a[1],
                    transform=ax.transAxes,
                    facecolor="none",
                    edgecolor="none",
                    gid=gid,
                    zorder=7,
                )
            )
            r["icon_gid"] = gid
            r["label_gid"] = gid.replace("pareto_icon_", "pareto_label_")
            t.set_gid(r["label_gid"])
    return metadata


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


def render(analysis_root, output_dir):
    configure("efficiency")
    POINTS = json.loads(
        (Path(analysis_root) / "costs/pareto_figure_data.json").read_text()
    )["rows"]
    panels = [
        ("quant", "cost", True, "A   Quantitative: learning gain vs. tutoring cost"),
        ("verbal", "cost", True, "B   Verbal: learning gain vs. tutoring cost"),
        (
            "quant",
            "latency",
            False,
            "C   Quantitative: learning gain vs. AI reply time",
        ),
        ("verbal", "latency", False, "D   Verbal: learning gain vs. AI reply time"),
    ]
    nrows, ncols, figsize = 2, 2, (14.5, 9.7)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    # Preserve the plotting-area height while tightening panel and legend gaps.
    # Axis limits and the scientific coordinates are set by the shared renderer.
    fig.subplots_adjust(
        left=0.085 if ncols == 1 else 0.07,
        right=0.99,
        bottom=0.13,
        top=0.95,
        hspace=0.30,
        wspace=0.22,
    )
    meta = []
    for ax, (scope, metric, human, title) in zip(axes.flat, panels):
        rows = [
            r for r in POINTS if r["scope"] == scope and (r["kind"] == "ai" or human)
        ]
        panel_meta = pareto_panel(ax, rows, metric, title)
        if scope == "quant" and metric == "cost":
            # This crowded pair needs a little extra horizontal separation;
            # keep both the model coordinate and every other label fixed.
            label = next(
                t for t in ax.texts if t.get_text().strip() == "Gemini 3.7 Flash (med)"
            )
            row = next(
                r for r in panel_meta if r["arm_id"] == "gemini-3.7-flash-medium"
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            pixel = fig.dpi / 72
            origin = ax.transData.transform(label.get_position())
            boxes = [t.get_window_extent(renderer) for t in ax.texts if t is not label]
            points = [
                matplotlib.transforms.Bbox.from_bounds(x - 6, y - 6, 12, 12)
                for x, y in ax.transData.transform(
                    [[r["x"], r["gain_pp"]] for r in panel_meta]
                )
            ]
            for offset in [12, 16, 20, 24, 30]:
                label.set_position(
                    ax.transData.inverted().transform(origin + [-offset * pixel, 0])
                )
                box = label.get_window_extent(renderer)
                if ax.bbox.contains(box.x0, box.y0) and not any(
                    box.overlaps(b) for b in boxes + points
                ):
                    break
            else:
                raise AssertionError(
                    "No clear position for Quantitative Gemini 3.7 Flash label"
                )
            row.update(
                label_xy=list(map(float, label.get_position())),
                label_overlaps=0,
                marker_overlaps=0,
                label_outside_px=0.0,
            )
        meta.extend(panel_meta)
        ax.set_ylabel("Learning gain (percentage points)", fontsize=11.5)
        ax.set_xlabel(
            (
                "Tutoring cost per session (USD; log scale)"
                if metric == "cost"
                else "AI reply time during tutoring (seconds; log scale)"
            ),
            fontsize=11.5,
        )
        if metric == "cost":
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
        axis_style(ax)
        ax.title.set_fontsize(13)
    handles = [
        plt.Line2D(
            [],
            [],
            color=HUMAN_RED,
            ls="",
            marker="*",
            markeredgecolor="white",
            markeredgewidth=1.05,
            markersize=11,
            label="Human tutor ($75/hour reference)",
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.018),
        ncol=1,
        frameon=False,
        fontsize=11,
    )
    # The manuscript places the two lower panels 8 points below the native grid.
    save_frontier_figure(
        fig,
        output_dir,
        "figure_17_section_frontiers",
        meta,
        panel_offsets={"axes_3": (0, 8), "axes_4": (0, 8)},
    )


if __name__ == "__main__":
    figure_cli(render)
