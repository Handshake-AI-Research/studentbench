"""Table G.2: expert rankings with reviewer and pre-test clustering."""

from pathlib import Path
from studentbench.table_output import render, cli, read_json


def run(data_dir, analysis_dir, output_dir):
    data = read_json(Path(analysis_dir) / "reviewer/results.json")["results"]
    rows = []
    for name, label in [
        ("planning_combined", "Lesson planning"),
        ("practice_combined", "Practice creation/design"),
        ("common8_combined", "All eight shared criteria"),
    ]:
        r = data[name]
        rows.append(
            {
                "Evaluation": label,
                "Non-tied ratings": r["n_decisive"],
                "Pre-tests": r["cluster_counts"]["pretest"],
                "Original": str(r["original"]["leader_ahead_count_holm"]) + "/12",
                "Sensitivity": str(r["sensitivity"]["leader_ahead_count_holm"]) + "/12",
            }
        )
    return render(
        output_dir,
        "table_14_reviewer_dependence",
        rows,
        "Table G.2. Repeated-reviewer sensitivity",
        "Same fitted scores; uncertainty clustered by reviewer and pre-test, with Student-t reference (50 degrees of freedom).",
    )


if __name__ == "__main__":
    cli(run)
