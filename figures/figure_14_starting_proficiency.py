"""Figure C.3. AI minus no-tutor gains by starting proficiency."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from studentbench.plotting import (
    configure,
    MODEL_TEXT,
    save_figure,
    figure_cli,
)


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
    df = pd.read_csv(DATA / "proficiency_quartile_ai_control_contrasts.csv")
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.6))
    fig.subplots_adjust(left=0.15, right=0.985, top=0.83, bottom=0.19, wspace=0.55)
    for i, sec in enumerate(["quant", "verbal"]):
        ax = axs[i]
        d = df[df.section == sec]
        ax.axvline(0, color="#aeb8bf", lw=0.7)
        for est, off, c, marker in [
            ("raw Welch", -0.12, "#86939d", "o"),
            ("ANCOVA HC3", 0.12, "#087ca7", "s"),
        ]:
            b = d[d.estimator == est].sort_values("quartile")
            y = np.arange(4) + off
            ax.errorbar(
                b.ai_minus_control_pp,
                y,
                xerr=[
                    b.ai_minus_control_pp - b.ci_low_pp,
                    b.ci_high_pp - b.ai_minus_control_pp,
                ],
                fmt=marker,
                color=c,
                elinewidth=1.4,
                capsize=3,
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=1.05,
                label="Observed" if est == "raw Welch" else "Baseline-adjusted",
                zorder=3,
            )
        b = d[d.estimator == "ANCOVA HC3"].sort_values("quartile")
        ax.set_yticks(
            range(4), [f"{r.quartile}  {r.score_label}" for r in b.itertuples()]
        )
        ax.set_ylim(3.55, -0.55)
        ax.set_title(f"{chr(65 + i)}  {sec.title()}", loc="left")
        ax.set_xlabel("AI − control learning gain (percentage points)")
        axes_base(ax)
        ax.set_xlim(
            min(-5, float(df.ci_low_pp.min()) - 1), float(df.ci_high_pp.max()) + 1
        )
    axs[1].legend(frameon=False, loc="lower right", fontsize=10)
    save_figure(fig, output_dir, "figure_14_starting_proficiency", pad_inches=0.09)


if __name__ == "__main__":
    figure_cli(render)
