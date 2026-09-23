"""Figure E.4. Quantitative AI tutor, domain and starting-proficiency differences."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from studentbench.plotting import (
    configure,
    MODEL_TEXT,
    DOMAIN_ORDER,
    QUARTILE_IDS,
    display_label,
    model_family,
    save_figure,
    figure_cli,
)


def render(analysis_root, output_dir):
    configure("learning")
    DATA = Path(analysis_root) / "learning"
    d = pd.read_csv(DATA / "topic_proficiency_configuration_effects.csv")
    d = d[d.section == "quant"]
    models = sorted(
        d.arm_id.unique(), key=lambda a: (model_family(a), display_label(a))
    )
    fig, axs = plt.subplots(1, 4, figsize=(12.6, 6.8), sharey=True)
    fig.subplots_adjust(left=0.205, right=0.965, top=0.84, bottom=0.23, wspace=0.10)
    lim = float(
        np.ceil(
            max(
                abs(d.ancova_ai_minus_control_pp.min()),
                abs(d.ancova_ai_minus_control_pp.max()),
            )
            / 5
        )
        * 5
    )
    for i, topic in enumerate([x[1] for x in DOMAIN_ORDER if x[0] == "quant"]):
        ax = axs[i]
        b = (
            d[d.topic == topic]
            .pivot(
                index="arm_id", columns="quartile", values="ancova_ai_minus_control_pp"
            )
            .loc[models, list(QUARTILE_IDS)]
        )
        im = ax.imshow(
            b.values,
            cmap="RdBu",
            norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim),
            aspect="auto",
        )
        ax.set_xticks(range(4), ["Q1", "Q2", "Q3", "Q4"])
        ax.set_yticks(range(len(models)), [display_label(a) for a in models])
        ax.tick_params(length=0)
        ax.tick_params(axis="y", labelcolor=MODEL_TEXT)
        ax.set_title(
            f"{chr(65 + i)}  {topic[:1] + topic[1:].lower()}",
            loc="left",
            fontsize=12,
            pad=10,
        )
        for y, a in enumerate(models):
            for x, q in enumerate(QUARTILE_IDS):
                r = d[(d.topic == topic) & (d.arm_id == a) & (d.quartile == q)].iloc[0]
                v = r.ancova_ai_minus_control_pp
                txt = f"{v:.0f}" + ("†" if bool(r.small_cell_n_lt_10) else "")
                ax.text(
                    x,
                    y,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(v) > lim * 0.55 else "#273039",
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Pre-test quartile", fontsize=10)
    cax = fig.add_axes([0.40, 0.115, 0.36, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("AI − control learning gain (percentage points)", fontsize=10)
    fig.text(
        0.205,
        0.94,
        "Quantitative teaching varies with subject matter and starting proficiency",
        fontsize=14,
        weight="bold",
    )
    tests = json.loads(
        (DATA / "topic_proficiency_configuration_omnibus.json").read_text()
    )
    p = next(
        t["p_holm_across_quant_verbal_global_frontiers"]
        for t in tests
        if t["section"] == "quant"
    )
    fig.text(
        0.205,
        0.885,
        "AI tutor × domain × quartile interaction: "
        + ("p<.001" if p < 0.001 else f"p = {p:.3f}".replace("0.", ".")),
        fontsize=11,
        color=MODEL_TEXT,
    )
    save_figure(fig, output_dir, "figure_15_domain_proficiency", pad_inches=0.09)


if __name__ == "__main__":
    figure_cli(render)
