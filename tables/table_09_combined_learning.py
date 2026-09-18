"""Table 9: observed and adjusted Combined learning gains."""

from pathlib import Path
import pandas as pd
from studentbench.table_output import render, cli, interval
from studentbench.plotting import display_label


def run(data_dir, analysis_dir, output_dir):
    root = Path(analysis_dir) / "learning"
    raw = pd.read_csv(root / "raw_arm_outcomes.csv")
    adjusted = pd.read_csv(root / "ancova_adjusted_arm_outcomes.csv")
    raw = raw.loc[raw.scope.eq("combined")]
    adjusted = adjusted.loc[adjusted.scope.eq("combined")].set_index("arm_id")
    rows = []
    ordered = list(
        raw.loc[raw.kind.eq("ai")].sort_values("mean_gain_pp", ascending=False).arm_id
    ) + ["human", "control"]
    for arm in ordered:
        r = raw.loc[raw.arm_id.eq(arm)].iloc[0]
        a = adjusted.loc[arm]
        rows.append(
            {
                "Tutor": display_label(arm),
                "n": int(r["n"]),
                "Observed gain [95% CI]": interval(
                    r.mean_gain_pp, r.ci_low_pp, r.ci_high_pp
                ),
                "Adjusted gain [95% CI]": interval(
                    a.adjusted_marginal_gain_pp, a.ci_low_pp, a.ci_high_pp
                ),
            }
        )
    return render(
        output_dir,
        "table_09_combined_learning",
        rows,
        "Table 9. Combined learning gains",
        "Observed Student-t intervals; adjusted HC3 intervals.",
    )


if __name__ == "__main__":
    cli(run)
