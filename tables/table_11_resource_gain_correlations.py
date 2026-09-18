"""Table 11: correlations of tutor-level resources with learning gain."""

from pathlib import Path
import pandas as pd
from studentbench.table_output import render, cli, pvalue, SECTIONS
import numpy as np
from scipy import stats


def run(data_dir, analysis_dir, output_dir):
    frame = pd.read_csv(Path(analysis_dir) / "costs/pareto_table.csv")
    rows = []
    predictors = [
        ("log10 cost", "mean_cost_usd", True),
        ("log10 reply time", "median_latency_s", True),
        ("Student messages", "mean_student_messages", False),
    ]
    for label, column, log in predictors:
        for scope in ["quant", "verbal", "combined"]:
            group = frame.loc[frame.scope.eq(scope) & frame.kind.eq("ai")]
            x = group[column].to_numpy()
            y = group.mean_gain_pp.to_numpy()
            pearson = stats.pearsonr(np.log10(x) if log else x, y)
            spearman = stats.spearmanr(x, y)
            rows.append(
                {
                    "Predictor": label,
                    "Scope": SECTIONS[scope],
                    "k": len(group),
                    "Pearson r": f"{pearson.statistic:.3f}",
                    "Pearson p": pvalue(pearson.pvalue),
                    "Spearman rho": f"{spearman.statistic:.3f}",
                    "Spearman p": pvalue(spearman.pvalue),
                }
            )
    return render(
        output_dir,
        "table_11_resource_gain_correlations",
        rows,
        "Table 11. Resources and gain across AI tutors",
        "One observation per AI tutor; per-comparison exploratory correlations.",
    )


if __name__ == "__main__":
    cli(run)
