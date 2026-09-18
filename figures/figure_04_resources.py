"""Figure 4. Tutoring cost, AI reply time and student engagement."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from contextlib import contextmanager
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import PathPatch, ConnectionPatch
from matplotlib.path import Path as PlotPath
import numpy as np
import pandas as pd
from scipy import stats
from studentbench.plotting import (
    configure,
    PARETO_BLUE,
    PARETO_TEXT,
    PARETO_SLATE,
    HUMAN_RED,
    COLORS,
    display_label,
    save_frontier_figure,
    figure_cli,
)

PANEL_TITLE_SIZE = 14


def pareto_panel(ax, rows, metric, title, zoom=False):
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
    ax.set_xscale("linear")
    ax.set_xlim(
        0, 5.0 if zoom else 80.0 if metric == "cost" else float(xs.max()) * 1.035
    )
    ax.xaxis.set_major_locator(
        ticker.MultipleLocator(1 if zoom else 10 if metric == "cost" else 5)
    )
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            (lambda v, _: f"${v:g}") if metric == "cost" else (lambda v, _: f"{v:g}")
        )
    )
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    span = max(float(np.ptp(ys)), 2.5)
    pad = max(span * 0.30, 1.0)
    # Retain lower padding; reduce visible upper headroom by about 40%.
    ax.set_ylim(float(ys.min()) - pad, float(ys.max()) + 0.55 * pad)
    if metric == "latency":
        ax.set_ylim(float(ys.min()) - 0.48, float(ys.max()) + 0.55)
    ax.yaxis.set_major_locator(ticker.FixedLocator([12, 13.5, 15, 16.5]))
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.grid(color="#e6eaed", lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#aeb8bf")
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(axis="both", labelsize=12, color="#aeb8bf", labelcolor="#38444c")
    ax.set_title(title, loc="left", weight="bold", fontsize=PANEL_TITLE_SIZE, pad=12)
    ax.set_ylabel("Observed gain (percentage points)", fontsize=13, labelpad=9)
    ax.set_xlabel(
        {
            "cost": "Estimated session cost (USD; logarithmic scale)",
            "latency": "Full reply time (seconds; logarithmic scale)",
            "engagement": "Student messages per session",
        }[metric],
        fontsize=13,
        labelpad=9,
    )
    # Extend the decorative wash to the axes boundary, then fade it to white
    # in SVG. This avoids a false vertical boundary at the final frontier point.
    # The wash is not an uncertainty interval or an additional frontier segment.
    ax.fill_between(
        [*[r[key] for r in frontier], ax.get_xlim()[1]],
        [*[r["mean_gain_pp"] for r in frontier], frontier[-1]["mean_gain_pp"]],
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
            fontsize=7.4 if zoom else 9.0,
            fontweight="normal",
            color=(
                "#202d36"
                if on_frontier
                else (
                    "#7f2a21"
                    if human
                    else "#697681"
                    if metric == "cost" and not zoom
                    else PARETO_TEXT
                )
            ),
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
        if (
            metric == "cost"
            and not zoom
            and r["x"] < 5
            and r["arm_id"]
            not in {
                "gemini-3.5-flash-low",
                "opus-4.8-xhigh",
                "opus-5-high",
                "gpt-5.5-high",
                "opus-4.8-off",
                "gemini-3.1-pro-high",
            }
        ):
            t.set_visible(False)
            r["label_shown"] = False
            continue
        r["label_shown"] = True
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
        if zoom:
            manual = {
                "gpt-5.4-mini-none": (1, 0, "left", "center", 0),
                "gemma-4-31b-high": (0.43, -1, "left", "top", 9),
                "kimi-k2.6": (1, 1, "left", "center", 2),
                "gemini-3.6-flash-low": (1, 0, "left", "top", 0),
                "opus-4.8-xhigh": (-0.5, 1, "right", "bottom", 6),
                "opus-4.8-off": (-1, 0, "right", "center", 0),
                "opus-5-high": (-0.5, -1, "right", "top", 7),
            }
            if r["arm_id"] in manual:
                dx, dy, ha, va, d = manual[r["arm_id"]]
                alignments = [(dx, dy, ha, va)]
                distances = [d]
        if metric == "latency" and not zoom:
            if r["arm_id"] == "gpt-5.4-mini-none":
                alignments = [(1, 0, "left", "center")]
                distances = [0]
            elif r["arm_id"] == "gemini-3.7-flash-medium":
                alignments = [(1, 1, "left", "bottom")]
                distances = [7]
            elif r["arm_id"] == "gemini-3.5-flash-low":
                alignments = [(1, 1, "left", "bottom")]
                distances = [7]
            elif r["arm_id"] == "gemini-3.1-pro-high":
                alignments = [(1, 1, "left", "bottom")]
                distances = [7]
            elif r["arm_id"] == "opus-4.8-xhigh":
                alignments = [(1, 1, "left", "bottom")]
                distances = [7]
            elif r["arm_id"] == "gemini-3.6-flash-low":
                alignments = [(-0.5, -1, "left", "top")]
                distances = [14]
            elif r["arm_id"] == "gpt-5.5-high":
                alignments = [(0, -1, "center", "top")]
                distances = [7]
            elif r["arm_id"] == "gemma-4-31b-high":
                alignments = [(1, -1, "left", "top")]
                distances = [7]
            elif r["arm_id"] == "opus-4.8-off":
                alignments = [(1, 0, "left", "center")]
                distances = [0]
            elif r["arm_id"] == "opus-5-high":
                alignments = [(1, -1, "left", "top")]
                distances = [7]
            elif r["arm_id"] == "kimi-k2.6":
                alignments = [(1, 1, "left", "bottom")]
                distances = [7]
            elif r["arm_id"] == "gpt-5.5-pro-med":
                alignments = [(-1, 0, "right", "center")]
                distances = [0]
        if metric == "cost" and not zoom:
            if r["human_reference"]:
                alignments = [(0, 1, "center", "bottom")]
                distances = [7]
            elif r["arm_id"] == "gpt-5.5-pro-med":
                alignments = [(-1, 0, "right", "center")]
                distances = [0]
            else:
                alignments = [(1, 0, "left", "center")]
                distances = [0]
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


@contextmanager
def row_journal(path):
    """Checkpoint each visible model coordinate while the figure is built."""
    with path.open("w") as stream:

        def append(row):
            stream.write(json.dumps(row) + "\n")
            stream.flush()

        yield append


def render(analysis_root, output_dir):
    """Recreate the paper's linear axes, Pareto zoom and adjusted engagement curve."""
    configure("efficiency")
    root = Path(analysis_root) / "costs"
    REPORT = Path(output_dir)
    REPORT.mkdir(parents=True, exist_ok=True)
    data = json.loads((root / "pareto_figure_data.json").read_text())
    rows = [row for row in data["rows"] if row["scope"] == "combined"]
    engagement = pd.read_csv(
        root / "latency_student_engagement_by_arm_scope.csv"
    ).query("scope == 'combined'")
    statistic = json.loads((root / "statistics.json").read_text())[
        "latency_student_engagement"
    ]["associations"]["combined"]
    engagement_stat = {
        "estimate": statistic["spearman_rho"],
        "p_holm_6": statistic["spearman_p_holm_6"],
    }
    engagement_proof = {}
    fit_x = engagement.median_latency_s.to_numpy(dtype=float)
    fit_y = engagement.mean_student_messages.to_numpy(dtype=float)
    fit_z = np.log10(fit_x)
    fit_slope, fit_intercept = np.polyfit(fit_z, fit_y, 1)
    fit_df = len(fit_x) - 2
    fit_residual_variance = float(
        np.sum((fit_y - (fit_intercept + fit_slope * fit_z)) ** 2) / fit_df
    )
    fit_sxx = float(np.sum((fit_z - fit_z.mean()) ** 2))
    fit_grid = np.linspace(float(fit_x.min()), float(fit_x.max()), 200)
    fit_grid_z = np.log10(fit_grid)
    fit_mean = fit_intercept + fit_slope * fit_grid_z
    fit_mean_se = np.sqrt(
        fit_residual_variance
        * (1 / len(fit_x) + (fit_grid_z - fit_z.mean()) ** 2 / fit_sxx)
    )
    fit_critical = float(stats.t.ppf(0.975, fit_df))
    fit_low = fit_mean - fit_critical * fit_mean_se
    fit_high = fit_mean + fit_critical * fit_mean_se
    panel = pareto_panel
    plt.rcParams.update(
        {
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.family": "Arial",
            "font.size": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(10.5, 9.1))
    outer = fig.add_gridspec(
        2, 1, left=0.085, right=0.99, bottom=0.13, top=0.95, hspace=0.30
    )
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.013)
    axes = [fig.add_subplot(outer[0]), fig.add_subplot(bottom[0])]
    engagement_container = fig.add_subplot(bottom[1])
    meta = []
    for ax, metric, title in zip(
        axes,
        ["cost", "latency"],
        [
            "A   Pareto frontier: learning gain vs. tutoring cost",
            "B   Pareto frontier: learning gain vs. reply time",
        ],
    ):
        selected = [r for r in rows if metric == "cost" or r["kind"] == "ai"]
        meta.extend(panel(ax, selected, metric, title))
        ax.set_ylabel("Learning gain (percentage points)", fontsize=11.5)
        ax.set_xlabel(
            (
                "Tutoring cost per session (USD)"
                if metric == "cost"
                else "AI reply time during tutoring (seconds)"
            ),
            fontsize=11.5,
        )
        if metric == "latency":
            grid_end = next(
                r["median_latency_s"]
                for r in selected
                if r["arm_id"] == "gpt-5.5-pro-med"
            )
            assert grid_end == max(r["median_latency_s"] for r in selected)
            ax.grid(False, axis="y")
            ax.hlines(
                [
                    y
                    for y in ax.get_yticks()
                    if ax.get_ylim()[0] <= y <= ax.get_ylim()[1]
                ],
                0,
                grid_end,
                color="#e6eaed",
                linewidth=0.65,
                zorder=1,
                gid="reply_time_horizontal_grid",
            )
            ax.spines["bottom"].set_bounds(0, grid_end)
    inset = axes[0].inset_axes([0.32, 0.18, 0.43, 0.76])
    inset.set_zorder(10)
    zoomrows = [
        dict(r, scope="combined_zoom")
        for r in rows
        if r["kind"] == "ai" and r["mean_cost_usd"] < 5
    ]
    zoommeta = panel(inset, zoomrows, "cost", "", zoom=True)
    inset.set_xlabel("")
    inset.set_ylabel("")
    inset.tick_params(labelsize=8)
    inset.text(
        0.02,
        0.98,
        "Zoom",
        transform=inset.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="normal",
        color="black",
        zorder=12,
    )
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_color("#7F8C95")
        spine.set_linewidth(0.75)
    # Show the precise region enlarged by the inset, behind data marks and labels.
    zoom_x0, zoom_x1 = inset.get_xlim()
    zoom_y0, zoom_y1 = inset.get_ylim()
    # Leave small gaps in the $5 edge behind text, preserving plain, selectable
    # labels without adding new badges or moving any labels or scientific marks.
    edge_x_px = axes[0].transData.transform((zoom_x1, zoom_y0))[0]
    blocked = []
    for text in axes[0].texts:
        if not text.get_visible() or text.get_bbox_patch() is not None:
            continue
        box = text.get_window_extent(fig.canvas.get_renderer()).padded(2)
        if box.x0 < edge_x_px < box.x1:
            lo, hi = (
                axes[0]
                .transData.inverted()
                .transform([(edge_x_px, box.y0), (edge_x_px, box.y1)])[:, 1]
            )
            if hi > zoom_y0 and lo < zoom_y1:
                blocked.append((max(lo, zoom_y0), min(hi, zoom_y1)))
    vertices = [
        (zoom_x1, zoom_y0),
        (zoom_x0, zoom_y0),
        (zoom_x0, zoom_y1),
        (zoom_x1, zoom_y1),
    ]
    codes = [PlotPath.MOVETO, PlotPath.LINETO, PlotPath.LINETO, PlotPath.LINETO]
    last_y = zoom_y0
    for lo, hi in sorted(blocked) + [(zoom_y1, zoom_y1)]:
        if lo > last_y:
            vertices.extend([(zoom_x1, last_y), (zoom_x1, lo)])
            codes.extend([PlotPath.MOVETO, PlotPath.LINETO])
        last_y = max(last_y, hi)
    zoom_outline = PathPatch(
        PlotPath(vertices, codes),
        fill=False,
        edgecolor="#7F8C95",
        linewidth=0.75,
        zorder=1.4,
        clip_on=False,
    )
    zoom_outline.set_gid("zoom_source_outline")
    axes[0].add_patch(zoom_outline)
    for side, y, inset_y in [("bottom", zoom_y0, 0), ("top", zoom_y1, 1)]:
        connector = ConnectionPatch(
            xyA=(zoom_x1, y),
            coordsA=axes[0].transData,
            xyB=(0, inset_y),
            coordsB=inset.transAxes,
            color="#7F8C95",
            linewidth=0.9,
            linestyle=(0, (0.1, 4.0)),
            capstyle="round",
            zorder=1.4,
            clip_on=False,
        )
        connector.set_gid(f"zoom_connector_{side}")
        axes[0].add_artist(connector)

    # C uses the same model cohort and horizontal scale as B. Its margins are
    # allocated inside the existing half-row so A, B and the inset do not move.
    engagement_container.set_axis_off()
    engagement_container.set_title(
        "C   Correlation: engagement vs. reply time",
        loc="left",
        weight="bold",
        fontsize=PANEL_TITLE_SIZE,
        pad=12,
    )
    engagement_ax = engagement_container.inset_axes([0.13, 0, 0.87, 1])
    engagement_ax.set_xlim(axes[1].get_xlim())
    # Retain every part of the confidence band within the observed fit domain.
    engagement_ax.set_ylim(
        min(10.5, float(fit_low.min()) - 0.6), max(38.5, float(fit_high.max()) + 0.6)
    )
    engagement_ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    engagement_ax.xaxis.set_minor_locator(ticker.NullLocator())
    engagement_ax.yaxis.set_major_locator(ticker.FixedLocator([10, 15, 20, 25, 30, 35]))
    engagement_ax.set_axisbelow(True)
    engagement_ax.grid(color="#e6eaed", lw=0.65)
    engagement_ax.spines[["top", "right"]].set_visible(False)
    for side in ["left", "bottom"]:
        engagement_ax.spines[side].set_color("#aeb8bf")
        engagement_ax.spines[side].set_linewidth(0.7)
    engagement_ax.tick_params(
        axis="both", labelsize=12, color="#aeb8bf", labelcolor="#38444c"
    )
    engagement_ax.set_xlabel(
        "AI reply time during tutoring (seconds)", fontsize=11.5, labelpad=9
    )
    engagement_ax.set_ylabel("Student messages per session", fontsize=11.5, labelpad=9)
    engagement_ax.spines["bottom"].set_bounds(
        0, float(engagement.median_latency_s.max())
    )
    engagement_ax.grid(False, axis="y")
    engagement_ax.hlines(
        [10, 15, 20, 25, 30, 35],
        0,
        float(engagement.median_latency_s.max()),
        color="#e6eaed",
        lw=0.65,
        zorder=1,
    )
    # Match the overall lightness of A's fading wash, rather than its darkest stop.
    engagement_ax.fill_between(
        fit_grid,
        fit_low,
        fit_high,
        color=PARETO_BLUE,
        alpha=0.035,
        linewidth=0,
        zorder=1.1,
        gid="engagement_fit_ci95",
    )
    engagement_ax.plot(
        fit_grid,
        fit_mean,
        color=PARETO_BLUE,
        lw=1.8,
        zorder=2,
        gid="engagement_fit_mean",
    )
    engagement_ax.text(
        0.98,
        0.975,
        f"Spearman ρ={engagement_stat['estimate']:.2f}\np={engagement_stat['p_holm_6']:.4f}".replace(
            "=0.", "=."
        ),
        transform=engagement_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#38444c",
        linespacing=1.35,
    )
    # Center nearby labels on their marks; short gray guides distinguish the
    # three closely spaced Flash/mini points from one another.
    label_offsets = {
        "gemini-3.1-pro-high": (17, 0, "left", "center"),
        "gemini-3.5-flash-low": (7, 0, "left", "center"),
        "gemini-3.6-flash-low": (16, 6.4, "left", "center"),
        "gemini-3.7-flash-medium": (14.308327, -0.14, "left", "center"),
        "gpt-5.4-mini-none": (19.646086, -0.9, "left", "center"),
        "gemma-4-31b-high": (7, 1, "left", "bottom"),
        "gpt-5.5-high": (-7, -7, "right", "top"),
        "opus-4.8-xhigh": (9, 5, "left", "center"),
        "opus-4.8-off": (-7, 0, "right", "center"),
        "opus-5-high": (7, -7, "left", "top"),
        "kimi-k2.6": (7, 6, "left", "bottom"),
        "gpt-5.5-pro-med": (-7, -7, "right", "top"),
    }
    engagement_texts = []
    engagement_metadata = []
    with row_journal(REPORT / "figure_04_plotted_rows.jsonl") as journal:
        for row in engagement.to_dict("records"):
            x, y = float(row["median_latency_s"]), float(row["mean_student_messages"])
            color = COLORS[row["family"]]
            engagement_ax.scatter(
                [x], [y], s=48, color=color, edgecolor="white", linewidth=1.05, zorder=4
            )
            dx, dy, ha, va = label_offsets[row["arm_id"]]
            if row["arm_id"] in {
                "gemini-3.6-flash-low",
                "gpt-5.4-mini-none",
                "gemini-3.7-flash-medium",
                "opus-4.8-xhigh",
            }:
                engagement_ax.annotate(
                    "",
                    (x, y),
                    xytext=(dx - 2, dy),
                    textcoords="offset points",
                    arrowprops=dict(
                        arrowstyle="-",
                        color="#86939d",
                        linewidth=0.65,
                        shrinkA=0,
                        shrinkB=4,
                    ),
                    zorder=3,
                )
            text = engagement_ax.annotate(
                display_label(row["arm_label"]),
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=9,
                color=PARETO_TEXT,
                zorder=5,
            )
            engagement_texts.append(text)
            record = {
                "arm_id": row["arm_id"],
                "name": display_label(row["arm_label"]),
                "x": x,
                "y": y,
                "n_students": row["n_students"],
                "family": row["family"],
                "label_offset_points": [dx, dy],
                "label_ha": ha,
                "label_va": va,
            }
            engagement_metadata.append(record)
            journal(record)
    fig.canvas.draw()
    engagement_renderer = fig.canvas.get_renderer()
    engagement_label_boxes = [
        t.get_window_extent(engagement_renderer).padded(1) for t in engagement_texts
    ]
    engagement_label_collisions = []
    for i, first in enumerate(engagement_label_boxes):
        for j, second in enumerate(engagement_label_boxes[:i]):
            if first.overlaps(second):
                engagement_label_collisions.append(
                    [engagement_metadata[i]["arm_id"], engagement_metadata[j]["arm_id"]]
                )
    engagement_proof.update(
        {
            "plotted_rows": engagement_metadata,
            "label_collisions": engagement_label_collisions,
            "axis_xlim": list(engagement_ax.get_xlim()),
            "axis_ylim": list(engagement_ax.get_ylim()),
        }
    )

    save_frontier_figure(
        fig,
        output_dir,
        "figure_04_resources",
        [row for row in meta + zoommeta if "icon_gid" in row],
        pad_inches=0.04,
        extra_canvas=(0, 0.727968),
    )


if __name__ == "__main__":
    figure_cli(render)
