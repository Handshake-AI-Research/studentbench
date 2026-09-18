"""Table 12: selected adjusted engagement, correct-practice, and gain associations."""

from pathlib import Path
from studentbench.table_output import render, cli, read_json, pvalue, SECTIONS


def run(data_dir, analysis_dir, output_dir):
    tests = read_json(Path(analysis_dir) / "engagement/results.json")["tests"]
    rows = []
    pairs = [
        ("Reply time / student messages", "latency_mean_s", "student_chat_messages"),
        (
            "Student messages / correct practice",
            "student_chat_messages",
            "practice_first_credit_closed",
        ),
        ("Correct practice / learning gain", "practice_first_credit_closed", "gain_pp"),
    ]
    digits = {("quant", 1): 5, ("verbal", 0): 2, ("verbal", 1): 3, ("verbal", 2): 4}
    for scope in ["quant", "verbal", "combined"]:
        for i, (label, x, y) in enumerate(pairs):
            r = next(
                r
                for r in tests
                if r["scope"] == scope
                and r["family"] == "primary_raw"
                and r["x"] == x
                and r["y"] == y
            )
            lo, hi = [v * 10 for v in r["ci95_per_unit"]]
            rows.append(
                {
                    "Section": SECTIONS[scope],
                    "Predictor / outcome": label,
                    "Beta per 10": f"{r['estimate_per_unit'] * 10:.2f}",
                    "95% CI": f"[{lo:.2f}, {hi:.2f}]",
                    "p": pvalue(r["p_holm276"], digits.get((scope, i), 3)),
                }
            )
    return render(
        output_dir,
        "table_12_engagement_practice",
        rows,
        "Table 12. Student engagement, correct practice and learning",
        "Normal-reference HC3 intervals; global Holm correction across all 276 tests.",
    )


if __name__ == "__main__":
    cli(run)
