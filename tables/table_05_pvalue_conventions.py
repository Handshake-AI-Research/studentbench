"""Table B.1: the paper's two statistical questions."""

from pathlib import Path
import json

from tables.output import render
from studentbench.table_output import cli

COUNT_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve",
)

ROWS = [
    {
        "Question": "Equivalence",
        "Reporting rule": "TOST at alpha=.05 with ±0.25-SD bounds; p is the larger "
        "of the two one-sided p values. Each AI tutor is tested separately against "
        "human tutoring, without correction across tutors; all {passing_tutors} passing tutors "
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
    path = Path(analysis_dir) / "costs/individual_model_equivalence.jsonl"
    with path.open() as stream:
        tests = [json.loads(line) for line in stream if line.strip()]
    tests = [test for test in tests if test["record_kind"] == "model_result"]
    if not tests:
        raise ValueError("Individual equivalence results are empty")
    count = sum(test["nominal_equivalent"] for test in tests)
    passing_tutors = COUNT_WORDS[count]
    rows = [dict(row) for row in ROWS]
    rows[0]["Reporting rule"] = rows[0]["Reporting rule"].format(
        passing_tutors=passing_tutors,
    )
    return render(
        output_dir,
        "table_05_pvalue_conventions",
        rows,
        "Table B.1. Two statistical questions",
        "Study reporting rules; the passing-tutor count comes from the individual CR2 equivalence tests.",
        values={"passing_tutors": passing_tutors},
    )


if __name__ == "__main__":
    cli(run)
