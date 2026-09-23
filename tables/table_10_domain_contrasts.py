"""Table C.2: seven domain contrasts against no tutoring."""

from pathlib import Path
import pandas as pd
from tables.output import render
from studentbench.table_output import cli, interval, KINDS
from studentbench.plotting import DOMAIN_ORDER


def run(data_dir, analysis_dir, output_dir):
    frame = pd.read_csv(Path(analysis_dir) / "learning/academic_field_effects.csv")
    rows = []
    for section, topic, _ in DOMAIN_ORDER:
        r = frame.loc[(frame.instrument == section) & (frame.topic == topic)].iloc[0]
        for kind in ["ai", "human"]:
            adj = "ancova_" + kind + "_minus_control_"
            rows.append(
                {
                    "Domain": topic.capitalize(),
                    "Condition": KINDS[kind],
                    "Gain relative to control [95% CI]": interval(
                        r[adj + "estimate_pp"],
                        r[adj + "ci_low_pp"],
                        r[adj + "ci_high_pp"],
                    ),
                }
            )
    return render(
        output_dir,
        "table_10_domain_contrasts",
        rows,
        "Table C.2. Domain learning contrasts",
        "Linear domain pre-test adjustment, with pointwise 95% HC3 intervals.",
    )


if __name__ == "__main__":
    cli(run)
