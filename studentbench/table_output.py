"""Consistent CSV, Markdown and LaTeX outputs for the paper's numbered tables."""

from pathlib import Path
import argparse
import csv
import json

SECTIONS = {"quant": "Quantitative", "verbal": "Verbal", "combined": "Combined"}
KINDS = {"ai": "AI", "human": "Human", "control": "No tutor"}


def pvalue(value, digits=3):
    return (
        "<.001"
        if float(value) < 0.001
        else f"{float(value):.{digits}f}".removeprefix("0")
    )


def interval(value, lower, upper, digits=1):
    return f"{float(value):.{digits}f} [{float(lower):.{digits}f}, {float(upper):.{digits}f}]"


def read_json(path):
    return json.loads(Path(path).read_text())


def render(output_dir, name, rows, title, source):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"{name}: empty table")
    fields = list(rows[0])
    with (output_dir / (name + ".csv")).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            handle.flush()

    def text(value):
        return str(value).replace("\n", " ").replace("|", "\\|")

    lines = [
        "# " + title,
        "",
        source,
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    lines += ["| " + " | ".join(text(row[k]) for k in fields) + " |" for row in rows]
    (output_dir / (name + ".md")).write_text("\n".join(lines) + "\n")

    def tex(value):
        text = str(value)
        for old, new in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
            text = text.replace(old, new)
        return text.replace("<", r"$<$").replace("→", r"$\to$")

    latex = [
        r"\begin{tabular}{" + "l" * len(fields) + "}",
        r"\toprule",
        " & ".join(tex(k) for k in fields) + r"\\",
        r"\midrule",
    ]
    latex += [" & ".join(tex(row[k]) for k in fields) + r"\\" for row in rows]
    latex += [r"\bottomrule", r"\end{tabular}"]
    (output_dir / (name + ".tex")).write_text("\n".join(latex) + "\n")
    return rows


def cli(run):
    parser = argparse.ArgumentParser(description=run.__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, default=Path("results/analysis"))
    parser.add_argument("--output", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    run(args.data, args.analysis_root, args.output)
