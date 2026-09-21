"""Table D.4: equivalence and engagement results after excluding repeat participants."""

from pathlib import Path
from studentbench.table_output import render, cli, read_json, pvalue, SECTIONS


def run(data_dir, analysis_dir, output_dir):
    data = read_json(Path(analysis_dir) / "sensitivity/results.json")
    rows = []
    for scope in ["combined", "quant"]:
        a, b = (
            data["equivalence"][scope]["original"],
            data["equivalence"][scope]["exclude_repeat"],
        )
        rows.append(
            {
                "Result": SECTIONS[scope] + " AI minus human gain",
                "Original estimate": f"{a['estimate']:.2f}",
                "Original p": pvalue(a["p"]),
                "After exclusion estimate": f"{b['estimate']:.2f}",
                "After exclusion p": pvalue(b["p"]),
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
        "Table D.4. Repeat-participation sensitivity",
        "Public exclusion membership; equivalence margins fixed to original data; Holm276 recomputed within each cohort.",
    )


if __name__ == "__main__":
    cli(run)
