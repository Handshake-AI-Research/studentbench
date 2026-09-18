"""Figure 19. The tutoring experience across AI tutors."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from studentbench.plotting import (
    configure,
    MODEL_TEXT,
    display_label,
    save_figure,
    figure_cli,
)
from studentbench.plotting import (
    SECTION_NAMES,
)

SCOPES = ["quant", "verbal", "combined"]
TITLES = SECTION_NAMES


def render(analysis_root, output_dir):
    configure("efficiency")
    root = Path(analysis_root) / "costs"
    ACT = pd.read_csv(root / "interaction_fingerprint_by_arm_scope.csv")
    STAT = json.loads((root / "statistics.json").read_text())
    b = ACT[ACT.scope == "combined"]
    panel = STAT["interaction_fingerprint"]["panels"]["combined"]
    metrics = [
        "latency",
        "student_messages",
        "ai_messages",
        "concepts_shown",
        "practice_shown",
        "practice_accuracy",
        "helpfulness",
    ]
    # Sort by conversational volume rather than arbitrary labels or a composite score.
    order = (
        b[b.metric == "student_messages"]
        .sort_values("display_value", ascending=False)
        .arm_id.tolist()
    )
    rank = np.array(
        [
            [b[(b.arm_id == a) & (b.metric == m)].iloc[0].rank_score for m in metrics]
            for a in order
        ]
    )
    val = np.array(
        [
            [
                b[(b.arm_id == a) & (b.metric == m)].iloc[0].display_value
                for m in metrics
            ]
            for a in order
        ]
    )
    fig, ax = plt.subplots(figsize=(11.9, 5.9))
    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.10, top=0.80)
    # Keep the categorical rank map and its exact cell colors as vector cells.
    ax.pcolormesh(
        np.arange(len(metrics) + 1) - 0.5,
        np.arange(len(order) + 1) - 0.5,
        rank,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        shading="flat",
        rasterized=False,
    )
    ax.set_xlim(-0.5, len(metrics) - 0.5)
    ax.set_ylim(len(order) - 0.5, -0.5)
    for i, a in enumerate(order):
        for j, m in enumerate(metrics):
            p = panel[m]
            leader = p["leader_arm_id"] == a
            v = val[i, j]
            label = (
                f"{v:.1f}%"
                if m == "practice_accuracy"
                else (
                    f"{v:.2f}"
                    if m in ["latency", "concepts_shown", "helpfulness"]
                    else f"{v:.1f}"
                )
            )
            if leader:
                label += f"\n{p['leader_resolved_wins']}/{p['leader_rivals']}"
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.48, i - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#112632",
                        linewidth=1.2,
                    )
                )
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=10.5,
                color="white" if rank[i, j] > 0.56 else "#1e303b",
                linespacing=1.0,
            )
    for x in [0.5, 2.5, 5.5]:
        ax.axvline(x, color="white", lw=3)
    ax.set_xticks(
        range(7),
        [
            "AI reply time\n(seconds)",
            "Student\nmessages",
            "AI tutor\nmessages",
            "Concepts\nshown",
            "Practice\nshown",
            "Practice credit\n(%)",
            "Helpfulness\n(1–5)",
        ],
        fontsize=11,
        color="black",
    )
    names = {r["arm_id"]: display_label(r["arm_label"]) for r in b.to_dict("records")}
    ax.set_yticks(range(12), [names[a] for a in order], fontsize=11, color=MODEL_TEXT)
    ax.tick_params(
        top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=8
    )
    ax.spines[:].set_visible(False)
    fig.suptitle(
        "The tutoring experience across AI tutors",
        x=0.015,
        y=0.995,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.015,
        0.94,
        "Combined sections · rows ordered by student-message volume · cells show observed values",
        fontsize=11,
        color=MODEL_TEXT,
    )
    for x, label in [(0, "Pace"), (1.5, "Dialogue"), (4, "Practice"), (6, "Rating")]:
        ax.text(
            x,
            1.15,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            fontsize=11.5,
            weight="bold",
            color="black",
        )
    fig.text(
        0.24,
        0.025,
        "Color: pale=last rank; dark=first rank within each column. Outlined cells: leader; resolved wins / rivals.",
        fontsize=10,
        color=MODEL_TEXT,
    )
    save_figure(fig, output_dir, "figure_19_tutoring_experience")


if __name__ == "__main__":
    figure_cli(render)
