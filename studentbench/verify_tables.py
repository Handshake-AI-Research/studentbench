"""Verify complete table contents and exact manuscript LaTeX fragments."""

from pathlib import Path
import csv
import json
import re
from .data import sha256
from .journal import Journal, write_json


def numbers(value):
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value).replace("−", "-")
    pattern = r"(?<![A-Za-z])[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
    return [float(x) for x in re.findall(pattern, text)]


def normalized_cell(value):
    """Normalize typography while retaining inequality, sign and missingness."""
    return " ".join(str(value).replace("−", "-").split())


def verify_cells(table_dir, expected_path, checked=None):
    """Check complete tables, including headers, row labels and operators."""
    expected = json.loads(Path(expected_path).read_text())
    actual_names = {path.name for path in Path(table_dir).glob("*.csv")}
    wanted_names = set(expected["tables"])
    if actual_names != wanted_names:
        raise ValueError(
            "Table inventory differs: "
            f"missing={sorted(wanted_names - actual_names)}, "
            f"unexpected={sorted(actual_names - wanted_names)}"
        )
    count = 0
    for name, target in expected["tables"].items():
        with (Path(table_dir) / name).open(newline="") as stream:
            actual = [[normalized_cell(cell) for cell in row] for row in csv.reader(stream)]
        wanted = [[normalized_cell(cell) for cell in row] for row in target]
        if actual != wanted:
            for row, (got, target) in enumerate(zip(actual, wanted)):
                if got != target:
                    raise ValueError(
                        f"Table content differs: {name} row {row}: {got!r} != {target!r}"
                    )
            raise ValueError(f"Table row count differs: {name}")
        count += sum(len(row) for row in actual)
        if checked is not None:
            checked(name)
    return count


def run(table_dir, output_dir, expected_path=None):
    table_dir = Path(table_dir)
    expected_path = (
        Path(expected_path) if expected_path else Path(__file__).resolve().parents[1]
        / "verification/expected_table_cells.json"
    )
    target = json.loads(expected_path.read_text())
    journal = Journal(Path(output_dir) / "checks.jsonl")
    signature = sha256(expected_path) + sha256(Path(__file__))
    pins = {}

    def record(name):
        pins[name] = sha256(table_dir / name)
        journal.save(name, signature + pins[name], {"passed": True, "file": name})

    structured_cells = verify_cells(table_dir, expected_path, checked=record)
    latex_path = expected_path.with_name("expected_table_latex.json")
    if not latex_path.exists():
        raise ValueError("Missing paper LaTeX table expectations")
    latex = json.loads(latex_path.read_text())
    if {p.name for p in table_dir.glob("*_paper.tex")} != set(latex["tables"]):
        raise ValueError("Paper LaTeX table inventory differs")
    for name, expected_hash in latex["tables"].items():
        if sha256(table_dir / name) != expected_hash:
            raise ValueError(f"Paper LaTeX table bytes differ: {name}")
        journal.save(name, sha256(latex_path) + expected_hash,
                     {"passed": True, "file": name})
    # Count printed numeric values; the complete-cell comparison above already
    # checks their exact values, signs, precision and inequality operators.
    numeric_cells = 0
    starts = target["numeric_columns_start"]
    for name, rows in target["tables"].items():
        number = str(int(name.split("_")[1]))
        if number in starts:
            numeric_cells += sum(len(numbers(cell)) for row in rows[1:]
                                 for cell in row[starts[number]:])
    receipt = dict(
        status="PASS", complete=True,
        empirical_tables=len(starts), protocol_tables=len(target["protocol_rows"]),
        numeric_cells=numeric_cells, structured_cells=structured_cells,
        byte_identical_latex_tables=len(latex["tables"]),
        input_sha256=pins, expected_sha256=sha256(expected_path),
        latex_expectations_sha256=sha256(latex_path), paper_commit=target["paper_commit"],
    )
    write_json(Path(output_dir) / "summary.json", receipt)
    return receipt
