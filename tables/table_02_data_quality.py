"""Table 2: primary exclusion counts at the two screening stages."""

from pathlib import Path
from studentbench.table_output import render, cli
from studentbench import data_quality


def run(data_dir, analysis_dir, output_dir):
    rows = data_quality.run(Path(data_dir), Path(analysis_dir) / "data_quality")
    rows = [
        {
            "Screening stage": r["stage"],
            "Primary exclusion reason": r["reason"],
            "Quant": r["quant"],
            "Verbal": r["verbal"],
            "Total": r["total"],
        }
        for r in rows
    ]
    return render(
        output_dir,
        "table_02_data_quality",
        rows,
        "Table 2. Data-quality exclusions",
        "Counts of the recorded primary reason; these are not overall study attrition rates.",
    )


if __name__ == "__main__":
    cli(run)
