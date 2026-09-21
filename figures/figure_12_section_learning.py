"""Figure E.1. Observed and adjusted gains by GRE section."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from studentbench.plotting import (
    configure,
    MODEL_TEXT,
    display_label,
    model_color,
    save_figure,
    write_json,
    figure_cli,
)


def ordered(frame, key):
    ai = frame[frame.kind == "ai"].sort_values([key, "arm_id"], ascending=[False, True])
    ref = pd.concat([frame[frame.kind == "human"], frame[frame.kind == "control"]])
    return pd.concat([ai, ref]).reset_index(drop=True)


def axes_base(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e6eaed", linewidth=0.65)
    for spine in ax.spines.values():
        spine.set_color("#aeb8bf")
        spine.set_linewidth(0.7)
    ax.tick_params(axis="y", length=0, pad=5, labelcolor=MODEL_TEXT)
    ax.tick_params(axis="x", length=3, color="#aeb8bf", labelcolor="#38444c")


def rank_panel(ax, d, key, title, model_order=None, show_model_labels=True):
    d = (
        ordered(d, key)
        if model_order is None
        else d.set_index("arm_id").loc[model_order].reset_index()
    )
    y = np.arange(len(d))
    lo = float(d.ci_low_pp.min()) - 0.9
    hi = float(d.ci_high_pp.max()) + 1.0
    axes_base(ax)
    ax.errorbar(
        d[key],
        y,
        xerr=[d[key] - d.ci_low_pp, d.ci_high_pp - d[key]],
        fmt="none",
        ecolor="#86939d",
        elinewidth=1.05,
        capsize=2,
        zorder=3,
    )
    for i, r in d.iterrows():
        ax.scatter(
            r[key],
            i,
            c=model_color(r.arm_id),
            s=100 if r.kind == "human" else 34,
            marker="*" if r.kind == "human" else "o",
            edgecolor="white",
            linewidth=1.05,
            zorder=4,
        )
    ax.set_yticks(y, [display_label(a) for a in d.arm_id])
    ax.set_ylim(len(d) - 0.5, -0.7)
    ax.set_xlim(lo, hi)
    ax.tick_params(axis="y", labelleft=show_model_labels)
    ax.axhline(12.5, color="#aeb8bf", lw=0.7)
    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel("Learning gain (percentage points)")
    ax.xaxis.set_major_locator(MaxNLocator(5))


def render(analysis_root, output_dir):
    configure("learning")
    DATA = Path(analysis_root) / "learning"
    raw = pd.read_csv(DATA / "raw_arm_outcomes.csv")
    adj = pd.read_csv(DATA / "ancova_adjusted_arm_outcomes.csv")
    # One model-label column per section keeps both estimates on the same row
    # and gives the released page width to the data rather than a second gutter.
    fig, axs = plt.subplots(2, 2, figsize=(12.6, 10.5))
    fig.subplots_adjust(
        left=0.20, right=0.985, top=0.95, bottom=0.07, wspace=0.10, hspace=0.27
    )
    plotted = []
    geometry = []
    labels = []
    for i, section in enumerate(["quant", "verbal"]):
        model_order = ordered(
            adj[adj.scope == section], "adjusted_marginal_gain_pp"
        ).arm_id.tolist()
        section_values = pd.concat(
            [raw[raw.scope == section], adj[adj.scope == section]]
        )
        limits = (
            float(section_values.ci_low_pp.min()) - 0.9,
            float(section_values.ci_high_pp.max()) + 1.0,
        )
        for j, (d, key, title) in enumerate(
            [
                (raw, "mean_gain_pp", "observed"),
                (adj, "adjusted_marginal_gain_pp", "baseline-adjusted"),
            ]
        ):
            rank_panel(
                axs[i, j],
                d[d.scope == section],
                key,
                f"{chr(65 + i * 2 + j)}  {section.title()} · {title}",
                model_order,
                j == 0,
            )
            axs[i, j].set_xlim(limits)
            axs[i, j].set_facecolor("none")
            block = d[d.scope == section].set_index("arm_id").loc[model_order]
            for y, (model, r) in enumerate(block.iterrows()):
                plotted.append(
                    {
                        "panel": chr(65 + i * 2 + j),
                        "section": section,
                        "model": model,
                        "y": y,
                        "x": float(r[key]),
                        "lo": float(r.ci_low_pp),
                        "hi": float(r.ci_high_pp),
                    }
                )
        # Keep the approved paired axes unchanged; extend Figure 3's light bands
        # through one left-aligned model column and both estimates.
        bounds = axs[i, 0].get_position()
        ylimits = axs[i, 0].get_ylim()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # Preserve the original tight crop and printed axis widths. Moving the
        # names into the outer figure margin would shrink both plots on the page.
        old_label_left = (
            min(t.get_window_extent(renderer).x0 for t in axs[i, 0].get_yticklabels())
            / fig.bbox.width
        )
        axs[i, 0].tick_params(axis="y", labelleft=False)
        from matplotlib.patches import Rectangle

        for y in np.arange(len(model_order))[::2]:
            edges = fig.transFigure.inverted().transform(
                axs[i, 0].transData.transform([[0, y - 0.5], [0, y + 0.5]])
            )[:, 1]
            band = Rectangle(
                (0.004, min(edges)),
                0.981,
                abs(edges[1] - edges[0]),
                transform=fig.transFigure,
                facecolor="#91afc3",
                alpha=0.1,
                linewidth=0,
                zorder=-1,
                clip_on=False,
            )
            band.set_in_layout(False)
            fig.add_artist(band)
        names = fig.add_axes(
            [old_label_left, bounds.y0, 0.195 - old_label_left, bounds.height]
        )
        names.set_ylim(ylimits)
        names.set_xlim(0, 1)
        names.axis("off")
        for y, model in enumerate(model_order):
            labels.append(
                names.text(
                    0,
                    y,
                    display_label(model),
                    ha="left",
                    va="center",
                    fontsize=10,
                    color="#7f2a21" if model == "human" else MODEL_TEXT,
                    clip_on=False,
                    gid=f"model_label_sections_{section}_{model}",
                )
            )
        names.axhline(12.5, color="#aeb8bf", linewidth=0.7, zorder=1)
        geometry.append(
            {
                "section": section,
                "model_order": model_order,
                "x_limits": list(limits),
                "axes_bounds": [list(a.get_position().bounds) for a in axs[i]],
                "original_label_left": old_label_left,
            }
        )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(renderer) for t in labels]
    for i, box in enumerate(boxes):
        assert box.x1 < axs[0, 0].bbox.x0 - 4, labels[i].get_text()
        assert fig.bbox.contains(box.x0, box.y0) and fig.bbox.contains(
            box.x1, box.y1
        ), labels[i].get_text()
        assert not any(box.overlaps(other) for other in boxes[:i]), labels[i].get_text()
    save_figure(fig, output_dir, "figure_12_section_learning", pad_inches=0.09)
    write_json(
        Path(output_dir) / "raw_adjusted_section_layout.json",
        {
            "status": "PASS",
            "style_reference": "main_teaching_capabilities",
            "geometry_preserved": True,
            "row_shading": "#91afc3",
            "row_shading_alpha": 0.1,
            "left_aligned_model_labels": True,
            "model_fontsize_pt": 10,
            "sections": geometry,
            "intervals": plotted,
        },
    )


if __name__ == "__main__":
    figure_cli(render)
