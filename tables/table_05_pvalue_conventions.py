"""Table B.1: the paper's two statistical questions."""

from tables.output import render
from studentbench.table_output import cli

ROWS = [
    {
        "Question": "Equivalence",
        "Reporting rule": "TOST at alpha=.05 with ±0.25-SD bounds; p is the larger "
        "of the two one-sided p values. Each AI tutor is tested separately against "
        "human tutoring, without correction across tutors; all four passing tutors "
        "are highlighted.",
    },
    {
        "Question": "Differences and associations",
        "Reporting rule": "Two-sided tests, or the stated omnibus test. For families "
        "of related comparisons, we report the Holm-corrected p value; the families "
        "and individual tests are specified in the paper's statistical methods.",
    },
]


def run(data_dir, analysis_dir, output_dir):
    return render(
        output_dir,
        "table_05_pvalue_conventions",
        ROWS,
        "Table B.1. Two statistical questions",
        "Study definitions; these rules specify the analyses rather than report fitted results.",
    )


if __name__ == "__main__":
    cli(run)
