"""Figure 1. Learning gains across GRE domains and AI tutors."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from studentbench.plotting import (
    configure,
    COLORS,
    HUMAN_RED,
    PARETO_BLUE,
    PARETO_SLATE,
    PARETO_TEXT,
    SECTION_COLORS,
    display_label,
    save_figure,
    figure_cli,
)


def learning_color(row):
    return (
        HUMAN_RED
        if row["kind"] == "human"
        else PARETO_SLATE
        if row["kind"] == "control"
        else COLORS[row["family"]]
    )


def learning_axes_style(ax, grid_axis):
    """Use the Figure 3A visual grammar without changing scientific scales."""
    for spine in ax.spines.values():
        spine.set_color("#aeb8bf")
        spine.set_linewidth(0.7)
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#e6eaed", lw=0.65)
    ax.tick_params(axis="both", color="#aeb8bf", labelcolor="#38444c")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")


def learning_name(row, compact=False):
    if row["kind"] == "human":
        return "Human tutor"
    if row["kind"] == "control":
        return "No tutor"
    return display_label(row["arm_label"])


def learning_rank_panel(ax, rows, *, compact=False, intervals=True, domain=False):
    """Horizontal stems use the labelled viewport; clipped CIs get arrows."""

    rows = list(rows)
    key = "mean_gain_pp" if domain else "adjusted_marginal_gain_pp"
    if compact:
        xmin, xmax = 5.0, 18.5
    elif intervals:
        xmin = math.floor(min(0, min(r["ci_low_pp"] for r in rows) - 0.4))
        xmax = math.ceil(max(r["ci_high_pp"] for r in rows) + 0.7)
    else:
        xmin = min(0, math.floor(min(r[key] for r in rows) - 0.7))
        xmax = math.ceil(max(r[key] for r in rows) + 2)
    fs = 9 if compact else 13.4
    for i, row in enumerate(rows):
        color = learning_color(row)
        value = row[key]
        ax.hlines(
            i,
            max(0, xmin),
            value,
            color=color,
            lw=4 if compact else 5,
            alpha=0.24,
            zorder=2,
        )
        if intervals:
            low, high = row["ci_low_pp"], row["ci_high_pp"]
            ax.hlines(
                i,
                max(low, xmin),
                min(high, xmax),
                color=color,
                lw=1 if compact else 1.25,
                zorder=3,
            )
            for end in [low, high]:
                if xmin <= end <= xmax:
                    ax.vlines(
                        end,
                        i - 0.11,
                        i + 0.11,
                        color=color,
                        lw=1 if compact else 1.1,
                        zorder=3,
                    )
            if low < xmin:
                ax.plot(
                    xmin + 0.045,
                    i,
                    marker="<",
                    ms=5,
                    color=color,
                    clip_on=False,
                    zorder=4,
                )
            if high > xmax:
                ax.plot(
                    xmax - 0.045,
                    i,
                    marker=">",
                    ms=5,
                    color=color,
                    clip_on=False,
                    zorder=4,
                )
        ax.scatter(
            value,
            i,
            color=color,
            s=155 if row["kind"] == "human" else 48 if compact else 56,
            marker="*" if row["kind"] == "human" else "o",
            edgecolor="white",
            linewidth=1.05,
            zorder=4,
        )
        ax.text(
            1.025,
            i,
            str(row["n"]),
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=fs if compact else fs - 0.3,
            color=PARETO_TEXT,
        )
    refs = [i for i, row in enumerate(rows) if row["kind"] != "ai"]
    if refs:
        ax.axhline(min(refs) - 0.52, color="#aeb8bf", lw=0.7)
    ax.set_yticks(
        range(len(rows)), [learning_name(r, compact) for r in rows], fontsize=fs
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(len(rows) - 0.6, -0.65)
    ax.set_xticks(
        [5, 10, 15, 18.5]
        if compact
        else np.arange(math.ceil(xmin / 5) * 5, xmax + 0.1, 5)
    )
    learning_axes_style(ax, "x")
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_color("#7f2a21" if compact and row["kind"] == "human" else PARETO_TEXT)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", labelsize=12 if compact else fs)
    if compact:
        ax.tick_params(axis="x", width=0.8, length=3.5, pad=3.5)
    ax.text(
        1.025,
        1.015,
        "n",
        transform=ax.transAxes,
        fontsize=fs,
        weight="normal" if compact else "bold",
    )
    ax.set_xlabel(
        (
            "Observed gain (percentage points)"
            if domain
            else "Adjusted gain (percentage points)"
        ),
        fontsize=11.5 if compact else fs,
        labelpad=9 if compact else 7,
    )
    return {
        "xlim": [xmin, xmax],
        "truncated_intervals": [
            {"arm_id": r["arm_id"], "lo": r["ci_low_pp"], "hi": r["ci_high_pp"]}
            for r in rows
            if intervals and (r["ci_low_pp"] < xmin or r["ci_high_pp"] > xmax)
        ],
    }


def learning_domains_panel(ax, domains):
    from matplotlib.lines import Line2D

    x = np.arange(7)
    ax.axvspan(-0.5, 3.5, color=SECTION_COLORS["quant"], alpha=0.06, zorder=0)
    ax.axvspan(3.5, 6.5, color=SECTION_COLORS["verbal"], alpha=0.06, zorder=0)
    ax.axvline(3.5, color="#aeb8bf", lw=0.7)
    styles = [
        ("ai", "#0072B2", "s", "-."),
        ("human", HUMAN_RED, "*", "-"),
        ("control", PARETO_SLATE, "o", "--"),
    ]
    for kind, color, marker, line in styles:
        y = np.array([r[kind]["mean_gain_pp"] for r in domains])
        lo = np.array([r[kind]["ci_low_pp"] for r in domains])
        hi = np.array([r[kind]["ci_high_pp"] for r in domains])
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            ls=line,
            lw=1.6,
            ms=11 if kind == "human" else 6,
            markeredgecolor="white",
            markeredgewidth=1.05,
            zorder=3,
        )
        ax.errorbar(
            x,
            y,
            yerr=[y - lo, hi - y],
            fmt="none",
            ecolor=color,
            alpha=0.43,
            capsize=2.6,
            lw=1,
            zorder=2,
        )
    maxima = [r["best_ai"]["mean_gain_pp"] for r in domains]
    ax.plot(x, maxima, ls=":", color=PARETO_SLATE, lw=1.3, zorder=2)
    offsets = [(0, 13), (0, 15), (0, 15), (12, 12), (12, 13), (0, 15), (0, 15)]
    for i, r in enumerate(domains):
        row = r["best_ai"]
        color = learning_color(row)
        ax.scatter(
            i,
            maxima[i],
            marker="D",
            s=56,
            color=color,
            edgecolor="white",
            lw=1.05,
            zorder=5,
        )
        label = learning_name(row, True).replace(" (", "\n(")
        ax.annotate(
            label,
            (i, maxima[i]),
            xytext=offsets[i],
            textcoords="offset points",
            ha="left" if i in [3, 4] else "center",
            va="bottom",
            fontsize=9,
            weight="normal",
            color=PARETO_TEXT,
        )
    labels = [
        "Data analysis",
        "Geometry",
        "Arithmetic",
        "Algebra",
        "Sentence\nequivalence",
        "Text\ncompletion",
        "Reading\ncomprehension",
    ]
    ax.set_xticks(
        x,
        [
            f"{label}\n({row['n_items']} questions)"
            for label, row in zip(labels, domains)
        ],
        fontsize=12,
    )
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-5.5, 36)
    ax.set_yticks([0, 10, 20, 30])
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Observed domain gain (percentage points)", fontsize=11.5, labelpad=9)
    learning_axes_style(ax, "y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", width=0.8, length=3.5, pad=3.5)
    ax.text(-0.37, 34.5, "Quantitative", va="top", fontsize=11.5, weight="bold")
    ax.text(3.62, 34.5, "Verbal", va="top", fontsize=11.5, weight="bold")
    handles = [
        Line2D(
            [0],
            [0],
            color=c,
            marker=m,
            ls=l,
            label=n,
            lw=1.5,
            markersize=11 if k == "human" else 6,
            markeredgecolor="white",
            markeredgewidth=1.05,
        )
        for (k, c, m, l), n in zip(styles, ["AI pooled", "Human tutor", "No tutor"])
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=PARETO_SLATE,
            marker="D",
            mfc="white",
            ls=":",
            label="Highest observed AI mean per topic",
        )
    )
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        ncol=2,
        fontsize=9.5,
        frameon=False,
        columnspacing=2.8,
        handlelength=2.2,
        borderaxespad=0,
    )


def learning_pooled_panel(ax, data):
    human = data["pooled_combined"]["human"]
    for bound in ["ci_low_pp", "ci_high_pp"]:
        ax.hlines(
            human[bound],
            -0.43,
            2.43,
            color=PARETO_SLATE,
            lw=1.15,
            ls=(0, (4, 3)),
            zorder=2,
        )
    for i, kind in enumerate(["ai", "human", "control"]):
        row = data["pooled_combined"][kind]
        c = {"ai": "#0072B2", "human": HUMAN_RED, "control": PARETO_SLATE}[kind]
        v = row["mean_gain_pp"]
        ax.bar(i, v, color=c, width=0.57, alpha=0.87, zorder=2)
        ax.errorbar(
            i,
            v,
            yerr=[[v - row["ci_low_pp"]], [row["ci_high_pp"] - v]],
            fmt="none",
            color="#86939d",
            capsize=2.4,
            capthick=1,
            lw=1,
            zorder=4,
        )
        ax.text(
            i,
            v / 2,
            f"{v:.1f}",
            ha="center",
            va="center",
            color="white",
            fontsize=12,
            weight="bold",
        )
    ax.set_xticks(
        range(3),
        [
            f"AI pooled\nn={data['pooled_combined']['ai']['n']:,}",
            f"Human tutor\nn={data['pooled_combined']['human']['n']}",
            f"No tutor\nn={data['pooled_combined']['control']['n']}",
        ],
        fontsize=12,
    )
    ax.set_ylim(0, 20)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Observed gain (percentage points)", fontsize=11.5, labelpad=9)
    learning_axes_style(ax, "y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", width=0.8, length=3.5, pad=3.5)


def render(analysis_root, output_dir):
    """Plot freshly reconstructed learning estimates, intervals and equivalence test."""
    configure("main")
    data = json.loads(
        (Path(analysis_root) / "learning/learning_figure_data.json").read_text()
    )
    # Compensate for the panels' different tight-crop margins so Figure 1
    # and Figure 3 retain the same type sizes at manuscript text width.
    fig = plt.figure(figsize=(10.83, 9.65))
    panel_title_size = 14
    domains = fig.add_axes([0.075, 0.595, 0.89, 0.30])
    learning_domains_panel(domains, data["domains"])
    fig.text(
        0.075,
        0.977,
        "A  AI teaching capability across academic domains",
        fontsize=panel_title_size,
        weight="bold",
        va="top",
    )
    pooled = fig.add_axes([0.075, 0.11, 0.335, 0.215])
    learning_pooled_panel(pooled, data)
    fig.text(
        0.075,
        0.475,
        "B  Combined Quantitative + Verbal",
        fontsize=panel_title_size,
        weight="bold",
        va="top",
    )
    fig.text(
        0.075,
        0.446,
        "Pooled AI and human gains are equivalent",
        fontsize=9.5,
        color=PARETO_BLUE,
        va="top",
    )
    tost = data["combined_tost"]
    ins = fig.add_axes([0.09, 0.385, 0.31, 0.045])
    margin = tost["equivalence_margin_pp"]
    ins.axvspan(-margin, margin, color=PARETO_BLUE, alpha=0.035)
    ins.axvline(0, color="#aeb8bf", lw=0.7)
    for edge in [-margin, margin]:
        ins.axvline(edge, color=PARETO_BLUE, lw=0.75, ls="--")
    lo = tost["equivalence_ci_low_pp"]
    hi = tost["equivalence_ci_high_pp"]
    v = tost["ai_minus_human_raw_gap_pp"]
    ins.errorbar(
        v,
        0,
        xerr=[[v - lo], [hi - v]],
        fmt="o",
        color=PARETO_BLUE,
        ms=5,
        markeredgecolor="white",
        markeredgewidth=1.05,
        capsize=3,
        lw=1.5,
    )
    ins.set_xlim(-5.2, 5.2)
    ins.set_ylim(-1, 1)
    ins.set_yticks([])
    ins.set_xticks([-margin, 0, margin], [f"−{margin:.2f}", "0", f"+{margin:.2f}"])
    ins.tick_params(labelsize=9)
    ins.spines[:].set_visible(False)
    ins.tick_params(
        color="#aeb8bf", labelcolor="#38444c", width=0.8, length=3.5, pad=3.5
    )
    fig.text(
        0.242,
        0.343,
        f"AI − human: 90% CI; TOST p={tost['p_tost']:.3f}".replace("=0.", "=."),
        ha="center",
        fontsize=9,
        color=PARETO_TEXT,
    )
    rank = fig.add_axes([0.625, 0.11, 0.315, 0.34])
    learning_rank_panel(rank, data["combined_adjusted"], compact=True)
    fig.text(
        0.46,
        0.475,
        "C  Learning by AI tutor",
        fontsize=panel_title_size,
        weight="bold",
        va="top",
    )
    save_figure(fig, output_dir, "figure_01_learning")


if __name__ == "__main__":
    figure_cli(render)
