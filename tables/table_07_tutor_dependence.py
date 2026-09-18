"""Table 7: equivalence with CR2 uncertainty for shared human tutors."""

from pathlib import Path
from studentbench.table_output import render, cli, read_json, pvalue, SECTIONS


def run(data_dir, analysis_dir, output_dir):
    values = read_json(Path(analysis_dir) / "tutor_dependence/results.json")["results"]
    rows = []
    for r in values:
        rows.append(
            {
                "Section": SECTIONS[r["section"]],
                "Difference": f"{r['estimate']:.2f}",
                "90% CI": f"[{r['ci90'][0]:.2f}, {r['ci90'][1]:.2f}]",
                "Bounds": f"±{r['margin']:.2f}",
                "p": pvalue(r["p"]),
            }
        )
    return render(
        output_dir,
        "table_07_tutor_dependence",
        rows,
        "Table 7. Shared-human-tutor sensitivity",
        "CR2 covariance with coefficient-specific Satterthwaite degrees of freedom.",
    )


if __name__ == "__main__":
    cli(run)
