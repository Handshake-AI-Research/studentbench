"""Figure C.2. AI tutor rankings across all seven GRE domains."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from studentbench.plotting import (
    configure,
    COLORS,
    MODEL_TEXT,
    DOMAIN_ORDER,
    display_label,
    model_color,
    save_figure,
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


def render(analysis_root, output_dir):
    configure("learning")
    DATA = Path(analysis_root) / "learning"
    data = pd.read_csv(DATA / "domain_arm_outcomes.csv")
    fig, axs = plt.subplots(4, 2, figsize=(12.6, 14.2))
    fig.subplots_adjust(
        left=0.20, right=0.97, top=0.969, bottom=0.035, wspace=1.04, hspace=0.29
    )
    for i, (section, topic, _) in enumerate(DOMAIN_ORDER):
        ax = axs.flat[i]
        d = ordered(data[data.topic == topic], "mean_gain_pp")
        y = np.arange(len(d))
        low = float(d.mean_gain_pp.min()) - 1
        high = float(d.mean_gain_pp.max()) + 2.3
        ax.barh(
            y,
            d.mean_gain_pp - low,
            left=low,
            height=0.62,
            color=[model_color(r["arm_id"]) for r in d.to_dict("records")],
            alpha=1,
            zorder=2,
        )
        ax.set_yticks(y, [display_label(a) for a in d.arm_id])
        ax.invert_yaxis()
        ax.set_xlim(low, high)
        ax.tick_params(axis="y", labelsize=10.8)
        ax.set_title(
            f"{chr(65 + i)}  {topic[:1] + topic[1:].lower()}", loc="left", pad=8
        )
        ax.axhline(12.5, color="#aeb8bf", lw=0.7)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        for j, r in d.iterrows():
            ax.text(
                r.mean_gain_pp + 0.27,
                j,
                f"{r.mean_gain_pp:.1f}",
                va="center",
                fontsize=9,
                color=MODEL_TEXT,
            )
        axes_base(ax)
    ax = axs.flat[7]
    ax.axis("off")
    ax.text(
        0,
        0.91,
        "Observed learning gain",
        weight="bold",
        fontsize=13,
        transform=ax.transAxes,
    )
    ax.text(0, 0.80, "Percentage points", fontsize=11, transform=ax.transAxes)
    ax.text(
        0,
        0.67,
        "AI tutors are ranked within each domain.\nHuman and control results are references.\nAxes begin 1 point below the smallest mean.",
        fontsize=10,
        linespacing=1.6,
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(
        handles=[
            Patch(
                color=COLORS[k],
                label={"human": "Human tutor", "control": "Control"}.get(k, k),
            )
            for k in ["OpenAI", "Anthropic", "Google", "Moonshot", "human", "control"]
        ],
        loc="lower left",
        frameon=False,
        ncol=2,
        fontsize=10,
        bbox_to_anchor=(-0.04, 0.02),
    )
    save_figure(fig, output_dir, "figure_13_domain_learning", pad_inches=0.09)


if __name__ == "__main__":
    figure_cli(render)
