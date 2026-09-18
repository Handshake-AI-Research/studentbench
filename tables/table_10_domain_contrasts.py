"""Table 10: seven domain contrasts against no tutoring."""

from pathlib import Path
import pandas as pd
from studentbench.table_output import render, cli, pvalue, interval, KINDS
from studentbench.plotting import DOMAIN_ORDER


def run(data_dir, analysis_dir, output_dir):
    frame = pd.read_csv(Path(analysis_dir) / "learning/academic_field_effects.csv")
    rows = []
    for section, topic, _ in DOMAIN_ORDER:
        r = frame.loc[(frame.instrument == section) & (frame.topic == topic)].iloc[0]
        for kind in ["ai", "human"]:
            raw = "welch_" + kind + "_minus_control_"
            adj = "ancova_" + kind + "_minus_control_"
            rows.append(
                {
                    "Domain": topic,
                    "Tutor": KINDS[kind],
                    "Raw difference [95% CI]": interval(
                        r[raw + "estimate_pp"],
                        r[raw + "ci_low_pp"],
                        r[raw + "ci_high_pp"],
                    ),
                    "Raw p": pvalue(r[raw + "p_two_sided"]),
                    "Adjusted difference [95% CI]": interval(
                        r[adj + "estimate_pp"],
                        r[adj + "ci_low_pp"],
                        r[adj + "ci_high_pp"],
                    ),
                    "Adjusted p": pvalue(r[adj + "p_two_sided_hc3"]),
                    "q": pvalue(r[adj + "p_two_sided_hc3_bh_within_instrument"]),
                }
            )
    return render(
        output_dir,
        "table_10_domain_contrasts",
        rows,
        "Table 10. Domain learning contrasts",
        "Raw Welch and linear-pretest-adjusted HC3 contrasts; q is within-section, within-tutor Benjamini–Hochberg correction.",
    )


if __name__ == "__main__":
    cli(run)
