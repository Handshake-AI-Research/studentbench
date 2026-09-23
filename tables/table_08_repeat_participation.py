"""Table B.3: equivalence and engagement results after excluding repeat participants."""

from pathlib import Path
from tables.output import render
from studentbench.table_output import cli, read_json, pvalue, SECTIONS


def run(data_dir, analysis_dir, output_dir):
    data = read_json(Path(analysis_dir) / "sensitivity/results.json")
    equivalence = read_json(Path(analysis_dir) / "primary_equivalence/results.json")[
        "models"
    ]
    rows = []
    for scope in ["combined", "quant"]:
        a = equivalence[f"adjusted_{scope}"]
        b = equivalence[f"exclude_repeat_adjusted_{scope}"]
        # The larger margin is the paper's common ±0.25-SD definition.
        a_test = max(a["margins"], key=lambda test: test["margin_pp"])
        b_test = max(b["margins"], key=lambda test: test["margin_pp"])
        rows.append(
            {
                "Result": SECTIONS[scope] + " AI − human gain",
                "Original estimate": f"{a['estimate_pp']:.2f}",
                "Original p": pvalue(a_test["p_tost"]),
                "After exclusion estimate": f"{b['estimate_pp']:.2f}",
                "After exclusion p": pvalue(b_test["p_tost"]),
            }
        )
    labels = [
        "Student messages per 10 seconds longer reply time",
        "Correct practice per 10 more student messages",
        "Gain per 10 more correct practice problems",
    ]
    pairs = [
        ("latency_mean_s", "student_chat_messages"),
        ("student_chat_messages", "practice_first_credit_closed"),
        ("practice_first_credit_closed", "gain_pp"),
    ]
    for label, (x, y) in zip(labels, pairs):
        a = next(
            r
            for r in data["original_selected"]
            if r["scope"] == "quant" and r["x"] == x and r["y"] == y
        )
        b = next(
            r
            for r in data["exclude_repeat_selected"]
            if r["scope"] == "quant" and r["x"] == x and r["y"] == y
        )
        rows.append(
            {
                "Result": label,
                "Original estimate": f"{a['estimate_per_unit'] * 10:.2f}",
                "Original p": pvalue(a["p_holm276"], 5),
                "After exclusion estimate": f"{b['estimate_per_unit'] * 10:.2f}",
                "After exclusion p": pvalue(b["p_holm276"], 5),
            }
        )
    return render(
        output_dir,
        "table_08_repeat_participation",
        rows,
        "Table B.3. Repeat-participation sensitivity",
        "Pooled CR2 equivalence with fixed original margins; engagement Holm correction recomputed across 276 tests within each cohort.",
    )


if __name__ == "__main__":
    cli(run)
