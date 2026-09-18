"""Table 1: starting scores and assessment-form order."""

from pathlib import Path
from studentbench.table_output import render, cli, SECTIONS, KINDS
from studentbench.data import load_sessions


def run(data_dir, analysis_dir, output_dir):
    frame = load_sessions(Path(data_dir), Path(analysis_dir) / "learning")
    rows = []
    for section in ["quant", "verbal"]:
        for kind in ["ai", "human", "control"]:
            group = frame.loc[(frame.section == section) & (frame.kind == kind)]
            rows.append(
                {
                    "Section": SECTIONS[section],
                    "Condition": KINDS[kind],
                    "Sessions": len(group),
                    "Pre-test, mean (SD)": f"{group.pre_pct.mean():.2f} ({group.pre_pct.std(ddof=1):.2f})",
                    "P→Q": int(group.form_order.eq("PQ").sum()),
                    "Q→P": int(group.form_order.eq("QP").sum()),
                }
            )
    return render(
        output_dir,
        "table_01_baseline",
        rows,
        "Table 1. Baseline scores and form order",
        "Calculated from released assessment responses.",
    )


if __name__ == "__main__":
    cli(run)
