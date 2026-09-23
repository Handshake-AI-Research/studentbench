"""Table D.1: correlations of tutor-level resources with learning gain."""

from pathlib import Path
import pandas as pd
from tables.output import render
from studentbench.table_output import cli, SECTIONS
from scipy import stats


def run(data_dir, analysis_dir, output_dir):
    frame = pd.read_csv(Path(analysis_dir) / "costs/pareto_table.csv")
    rows = []
    predictors = [
        ("Cost", "mean_cost_usd"),
        ("Reply time", "median_latency_s"),
        ("Student messages", "mean_student_messages"),
    ]
    for label, column in predictors:
        for scope in ["quant", "verbal", "combined"]:
            group = frame.loc[frame.scope.eq(scope) & frame.kind.eq("ai")]
            x = group[column].to_numpy()
            y = group.mean_gain_pp.to_numpy()
            spearman = stats.spearmanr(x, y)
            rows.append(
                {
                    "Predictor": label,
                    "Scope": SECTIONS[scope],
                    "k": len(group),
                    "Spearman rho": f"{spearman.statistic:.3f}",
                }
            )
    return render(
        output_dir,
        "table_11_resource_gain_correlations",
        rows,
        "Table D.1. Resources and gain across AI tutors",
        "One observation per AI tutor; Spearman correlations compare ranks.",
    )


if __name__ == "__main__":
    cli(run)
