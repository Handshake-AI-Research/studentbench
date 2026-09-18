"""Figure 6. AI reply time, student messages, correct practice and learning."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from studentbench.plotting import (
    configure,
    SECTION_COLORS,
    SECTION_LINE_COLORS,
    SECTION_TEXT_COLORS,
    figure_cli,
)
from matplotlib.lines import Line2D
from scipy import stats
import xml.etree.ElementTree as ET

SECTION = {
    "quant": {
        "label": "Quantitative",
        "color": SECTION_COLORS["quant"],
        "marker": "o",
        "ls": "-",
        "n": 1139,
    },
    "verbal": {
        "label": "Verbal",
        "color": SECTION_COLORS["verbal"],
        "marker": "s",
        "ls": (0, (4, 2.5)),
        "n": 1000,
    },
}
FIRST_ATTEMPT_PANELS = [
    {
        "x": "latency_mean_s",
        "y": "student_chat_messages",
        "title": "A  Student engagement vs.\n    AI reply time",
        "xlabel": "AI reply time (seconds)",
        "ylabel": "Student messages\n(per session)",
        "stat_corner": "right",
    },
    {
        "x": "student_chat_messages",
        "y": "first_closed_credit",
        "title": "B  Correct practice vs.\n    student engagement",
        "xlabel": "Student messages\n(per session)",
        "ylabel": "Correct practice (count)",
        "stat_corner": "right",
    },
    {
        "x": "first_closed_credit",
        "y": "gain_pp",
        "title": "C  Learning gain vs.\n    correct practice",
        "xlabel": "Correct practice (count)",
        "ylabel": "Learning gain (percentage points)",
        "stat_corner": "left",
    },
]


def fit_model(records, specification):
    """Fit the selected adjusted mean and its HC3 uncertainty for plotting."""
    data = [r for r in records if r["instrument"] == specification["scope"]]
    x = np.asarray([r[specification["x"]] for r in data], dtype=float)
    y = np.asarray([r[specification["y"]] for r in data], dtype=float)
    n = len(x)
    xbar, scale = float(x.mean()), float(x.std(ddof=1))
    columns = [np.ones(n), (x - xbar) / scale, np.asarray([r["pre_pct"] for r in data])]
    names = ["intercept", "standardized_exposure", "pre_pct"]
    for field in ("arm_id", "form_order"):
        for level in sorted({r[field] for r in data})[1:]:
            columns.append(np.asarray([r[field] == level for r in data], dtype=float))
            names.append(f"{field}:{level}")
    X = np.column_stack(columns)
    rank = np.linalg.matrix_rank(X)
    assert rank == X.shape[1] == specification["columns"]
    inverse = np.linalg.inv(X.T @ X)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, inverse, X)
    adjusted = residual / (1 - leverage)
    covariance = inverse @ (X.T @ ((adjusted**2)[:, None] * X)) @ inverse
    se = float(np.sqrt(covariance[1, 1]))
    z = float(stats.norm.ppf(0.975))
    ci = [float(beta[1] - z * se), float(beta[1] + z * se)]
    p = float(2 * stats.norm.sf(abs(beta[1] / se)))
    assert n == specification["n"]
    # Standardize nuisance covariates to their empirical section means. Because
    # the model is additive, this is equivalent to averaging predictions over
    # all sessions with exposure set to x. The fitted mean at xbar equals ybar.
    marginal_design = X.mean(axis=0)
    adjusted_y = y - (X[:, 2:] - marginal_design[2:]) @ beta[2:]
    assert np.allclose(
        adjusted_y, residual + beta[1] * X[:, 1] + float(y.mean()), atol=1e-9
    )
    # Stable ranks give ten groups of equal size +/-1, even with tied counts.
    # No session is omitted: extremes contribute to the first/last group.
    groups = np.array_split(np.argsort(x, kind="stable"), 10)
    bins = [
        {
            "group": k + 1,
            "n": len(idx),
            "x_mean": float(x[idx].mean()),
            "adjusted_y_mean": float(adjusted_y[idx].mean()),
            "x_min": float(x[idx].min()),
            "x_max": float(x[idx].max()),
        }
        for k, idx in enumerate(groups)
    ]
    assert sum(r["n"] for r in bins) == n
    # The display span is the exposure means, not the extrema of individual
    # sessions. This avoids extrapolating a marginal line across a sparse tail.
    grid = np.linspace(bins[0]["x_mean"], bins[-1]["x_mean"], 200)
    design = np.tile(marginal_design, (len(grid), 1))
    design[:, 1] = (grid - xbar) / scale
    mean = design @ beta
    mean_se = np.sqrt(
        np.maximum(0, np.einsum("ij,jk,ik->i", design, covariance, design))
    )
    multiplier = 10 / scale
    return {
        "record_id": specification["record_id"],
        "scope": specification["scope"],
        "x": specification["x"],
        "y": specification["y"],
        "n": n,
        "status": "complete",
        "n_valid": n,
        "rank": int(rank),
        "columns": X.shape[1],
        "exposure_scale": scale,
        "estimate": float(beta[1]),
        "se": se,
        "ci95": ci,
        "p": p,
        "estimate_per10": float(beta[1] * multiplier),
        "ci95_per10": [value * multiplier for value in ci],
        "x_mean": xbar,
        "y_mean": float(y.mean()),
        "x_full_range": [float(x.min()), float(x.max())],
        "coefficient_names": names,
        "coefficients": beta.tolist(),
        "hc3_covariance": covariance.tolist(),
        "marginal_design": marginal_design.tolist(),
        "bins": bins,
        "grid": grid.tolist(),
        "fit_mean": mean.tolist(),
        "fit_ci95_low": (mean - z * mean_se).tolist(),
        "fit_ci95_high": (mean + z * mean_se).tolist(),
    }


def p_label(value):
    """Use the main-paper threshold without altering stored probabilities."""
    if value < 0.001:
        return "$p$ < .001"
    return f"$p$={value:.3g}".replace("=0.", "=.")


def draw_panels(fits, output, panels, section_counts):
    section_counts = section_counts or {
        scope: style["n"] for scope, style in SECTION.items()
    }
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12.5,
            "axes.labelsize": 15.488084470,
            "axes.titlesize": 13.5,
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "#38444c",
            "ytick.color": "#38444c",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.4))
    fig.subplots_adjust(left=0.088, right=0.984, bottom=0.22, top=0.76, wspace=0.40)
    legend = [
        Line2D(
            [0],
            [0],
            color=SECTION_LINE_COLORS[scope],
            marker=s["marker"],
            linestyle=s["ls"],
            markerfacecolor=s["color"],
            markeredgecolor=SECTION_LINE_COLORS[scope],
            markersize=6,
            linewidth=2,
            label=f"{s['label']} (n={section_counts[scope]:,})",
        )
        for scope, s in SECTION.items()
    ]
    # Preserve the final manuscript legend spacing at its native 12.6-inch width.
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.53 - 3.46875 / (12.6 * 72), 0.995),
        frameon=False,
        ncol=2,
        handlelength=2.6,
        columnspacing=3.355,
        fontsize=12.5,
    )
    for ax, panel in zip(axes, panels):
        ax.set_title(
            panel["title"], loc="left", fontsize=16.613, pad=11, linespacing=1.25
        )
        ax.set_xlabel(panel["xlabel"], labelpad=8, linespacing=1.3)
        ax.set_ylabel(panel["ylabel"], labelpad=7, linespacing=1.3)
        ax.spines[["top", "right"]].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#aeb8bf")
            ax.spines[side].set_linewidth(0.7)
        ax.tick_params(labelsize=14.2, color="#aeb8bf", length=3)
        ax.set_axisbelow(True)
        ax.grid(color="#e6eaed", linewidth=0.65)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.margins(x=0.06, y=0.10)
        for scope, style in SECTION.items():
            fit = next(
                r
                for r in fits
                if r["scope"] == scope and r["x"] == panel["x"] and r["y"] == panel["y"]
            )
            x = np.asarray(fit["grid"])
            ax.fill_between(
                x,
                fit["fit_ci95_low"],
                fit["fit_ci95_high"],
                color=style["color"],
                alpha=0.12,
                linewidth=0,
                zorder=1,
                gid=f"{scope}_{panel['x']}_hc3_ci95",
            )
            ax.plot(
                x,
                fit["fit_mean"],
                color=SECTION_LINE_COLORS[scope],
                ls=style["ls"],
                linewidth=1.9,
                zorder=3,
                gid=f"{scope}_{panel['x']}_adjusted_mean",
            )
            ax.scatter(
                [b["x_mean"] for b in fit["bins"]],
                [b["adjusted_y_mean"] for b in fit["bins"]],
                c=style["color"],
                marker=style["marker"],
                s=25,
                edgecolors=SECTION_LINE_COLORS[scope] if scope == "verbal" else "white",
                linewidths=0.7,
                zorder=4,
                gid=f"{scope}_{panel['x']}_exposure_group_means",
            )
        # Show the chosen family's covariate-adjusted slope tests in each
        # panel. Effect sizes and their intervals remain in the results/table.
        right = panel["stat_corner"] in {"right", "bottom_right"}
        y_positions = (
            (0.19, 0.11) if panel["stat_corner"] == "bottom_right" else (0.95, 0.87)
        )
        for scope, y in zip(("quant", "verbal"), y_positions):
            fit = next(
                r
                for r in fits
                if r["scope"] == scope and r["x"] == panel["x"] and r["y"] == panel["y"]
            )
            name = "Quant" if scope == "quant" else "Verbal"
            # Non-threshold annotations retain the paper's two-space leading indent.
            # MathText encodes it as a text advance, without invisible space glyphs.
            ax.text(
                0.97 if right else 0.03,
                y,
                (r"$\hspace{0.6670574443071237}$" if fit["holm_p"] >= 0.001 else "")
                + f"{name}: Holm {p_label(fit['holm_p'])}",
                transform=ax.transAxes,
                ha="right" if right else "left",
                va="top",
                fontsize=10.65,
                color=SECTION_TEXT_COLORS[scope],
                zorder=6,
                gid=f"{scope}_{panel['x']}_holm_p",
            )
    for extension in ("pdf", "svg", "png"):
        target = output.with_suffix("." + extension)
        fig.savefig(
            target,
            dpi=180,
            bbox_inches="tight",
            pad_inches=0.08,
            metadata=(
                {
                    "Creator": "StudentBench",
                    "CreationDate": datetime(2026, 9, 17, 15, 8, 15),
                }
                if extension == "pdf"
                else None
            ),
        )
    plt.close(fig)
    # Match the publication PDF's lossless stream serialization. This changes no
    # drawing operators, text, estimates or font data.
    import pymupdf

    pdf = output.with_suffix(".pdf")
    temporary = output.with_suffix(".canonical.pdf")
    with pymupdf.open(pdf) as document:
        for reference in document[0].get_contents():
            document.update_stream(
                reference, document.xref_stream(reference), compress=True
            )
        document.save(temporary, garbage=0, deflate=False, no_new_id=True)
    temporary.replace(pdf)
    if panels == FIRST_ATTEMPT_PANELS:
        # This renderer applies the approved axis/title font sizes directly;
        # legacy font selectors identify the former tutor-turn labels.
        svg = output.with_suffix(".svg")
        root = ET.parse(svg).getroot()
        root.set("data-student-engagement-style", "student-engagement-v1")
        if panels == FIRST_ATTEMPT_PANELS:
            root.set("data-practice-definition", "first-closed-format-attempt-credit")
            root.set("data-holm-family-size", "276")
        svg.write_bytes(ET.tostring(root, encoding="utf-8"))


def render(analysis_root, output_dir):
    """Plot the six selected fits from the complete 276-test exploratory family."""
    configure("main")
    package = json.loads(
        (
            Path(analysis_root) / "engagement/first_attempt_practice_learning.json"
        ).read_text()
    )
    records = [
        row for row in package["session_inputs"] if row.get("common_primary", True)
    ]
    section_counts = {
        section: sum(row["instrument"] == section for row in records)
        for section in SECTION
    }
    fits = []
    for selected in package["selected_figure"]:
        specification = {**selected, "record_id": selected["source_test_id"]}
        fit = fit_model(records, specification)
        fit["holm_p"] = selected["p_holm276"]
        fits.append(fit)
    output = Path(output_dir) / "figure_06_engagement_practice"
    output.parent.mkdir(parents=True, exist_ok=True)
    draw_panels(fits, output, FIRST_ATTEMPT_PANELS, section_counts)


if __name__ == "__main__":
    figure_cli(render)
