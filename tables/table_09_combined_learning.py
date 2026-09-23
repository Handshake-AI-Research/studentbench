"""Table C.1: Combined learning gains adjusted for starting score and section."""

from pathlib import Path
import pandas as pd
from tables.output import render
from studentbench.table_output import cli, interval
from studentbench.plotting import display_label


def run(data_dir, analysis_dir, output_dir):
    root = Path(analysis_dir) / "learning"
    adjusted = pd.read_csv(root / "ancova_adjusted_arm_outcomes.csv")
    adjusted = adjusted.loc[adjusted.scope.eq("combined")]
    rows = []
    ordered = list(
        adjusted.loc[adjusted.kind.eq("ai")]
        .sort_values("adjusted_marginal_gain_pp", ascending=False).arm_id
    ) + ["human", "control"]
    for arm in ordered:
        a = adjusted.loc[adjusted.arm_id.eq(arm)].iloc[0]
        rows.append(
            {
                "Condition": {"control": "Control", "human": "Human"}.get(
                    arm, display_label(arm)
                ),
                "n": int(a["n"]),
                "Learning gain [95% CI]": interval(
                    a.adjusted_marginal_gain_pp, a.ci_low_pp, a.ci_high_pp
                ),
            }
        )
    return render(
        output_dir,
        "table_09_combined_learning",
        rows,
        "Table C.1. Combined learning gains",
        "Quadratic pre-test and section adjustment, with 95% HC3 intervals.",
    )


if __name__ == "__main__":
    cli(run)
