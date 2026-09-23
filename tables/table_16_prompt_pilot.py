"""Table G.1: all five exploratory prompt-pilot settings."""

from pathlib import Path
import pandas as pd
from tables.output import render
from studentbench.table_output import cli, read_json


def run(data_dir, analysis_dir, output_dir):
    frame = pd.read_csv(
        Path(analysis_dir) / "prompts/group_estimates.csv", keep_default_na=False
    )
    frame = frame.loc[frame.dataset.eq("pilot")]
    rows = []
    # The condition identifiers are the recorded lesson-plan/tutor settings.
    for condition in ["lean/lean", "lean/max", "mid/mid", "max/lean", "max/max"]:
        group = frame.loc[frame.condition.eq(condition)]
        row = {
            "Lesson plan / tutor": " / ".join(
                {"lean": "Minimal", "mid": "Intermediate", "max": "Expanded"}[v]
                for v in condition.split("/")
            )
        }
        for model in ["claude-opus-4-8", "gemini-3.5-flash", "gpt-5.5", ""]:
            subset = group.loc[group.model.eq(model)]
            if len(subset) != 1:
                raise ValueError((condition, model, len(subset)))
            r = subset.iloc[0]
            row[
                {
                    "claude-opus-4-8": "Opus 4.8",
                    "gemini-3.5-flash": "Gemini 3.5 Flash",
                    "gpt-5.5": "GPT-5.5",
                    "": "All",
                }[model]
            ] = f"{r.mean_gain_pp:.2f} ({int(r.n)})"
        rows.append(row)
    return render(
        output_dir,
        "table_16_prompt_pilot",
        rows,
        "Table G.1. Five prompt-pilot settings",
        "Fresh mean percentage-point gain and session counts; pilot sessions are separate from the main study.",
        values={
            "plan_tutor_mismatches": next(
                r["plan_tutor_identity_mismatches"]
                for r in read_json(Path(analysis_dir) / "prompts/summary.json")["coverage"]
                if r["dataset"] == "pilot"
            )
        },
    )


if __name__ == "__main__":
    cli(run)
