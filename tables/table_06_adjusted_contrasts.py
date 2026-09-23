"""Table B.2: adjusted tutoring-minus-no-tutor learning contrasts."""

from pathlib import Path
from tables.output import render
from studentbench.table_output import cli, read_json, pvalue, SECTIONS, KINDS


def run(data_dir, analysis_dir, output_dir):
    values = read_json(Path(analysis_dir) / "learning/primary_results.json")[
        "adjusted_contrasts"
    ]
    rows = []
    for scope in ["quant", "verbal", "combined"]:
        for kind in ["ai", "human"]:
            r = next(
                r
                for r in values
                if r["scope"] == scope
                and r["left_kind"] == kind
                and r["right_kind"] == "control"
            )
            rows.append(
                {
                    "Section": SECTIONS[scope],
                    "Contrast": KINDS[kind] + " − control",
                    "Estimate": f"{r['estimate_pp']:.2f}",
                    "95% CI": f"[{r['ci_low_pp']:.2f}, {r['ci_high_pp']:.2f}]",
                    "p": pvalue(
                        r["p_two_sided_hc3"],
                        4 if scope == "quant" and kind == "human" else 3,
                    ),
                }
            )
    return render(
        output_dir,
        "table_06_adjusted_contrasts",
        rows,
        "Table B.2. Adjusted learning contrasts",
        "HC3 intervals after adjustment for pre-test score, its square, and section in Combined.",
    )


if __name__ == "__main__":
    cli(run)
