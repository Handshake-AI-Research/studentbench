"""Verify complete table contents and every printed numeric result."""

from pathlib import Path
import csv
import json
import math
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


def verify_cells(table_dir, expected_path):
    """Check complete tables, including headers, row labels and operators."""
    expected = json.loads(Path(expected_path).read_text())
    count = 0
    for name, target in expected["tables"].items():
        with (Path(table_dir) / name).open(newline="") as stream:
            actual = [[normalized_cell(cell) for cell in row] for row in csv.reader(stream)]
        wanted = [[normalized_cell(cell) for cell in row] for row in target]
        if actual != wanted:
            raise ValueError(f"Table content differs, including labels/operators: {name}")
        count += sum(len(row) for row in actual)
    return count


def run(table_dir, output_dir, expected_path=None):
    table_dir = Path(table_dir)
    expected_path = (
        Path(expected_path)
        if expected_path
        else Path(__file__).resolve().parents[1]
        / "verification/expected_table_numbers.json"
    )
    target = json.loads(expected_path.read_text())
    cell_path = expected_path.with_name("expected_table_cells.json")
    if not cell_path.exists():
        raise ValueError("Missing complete table-cell expectations")
    structured_cells = verify_cells(table_dir, cell_path)
    actual = {}
    pins = {}
    for number, skip in target["skip_columns"].items():
        path = next(table_dir.glob(f"table_{int(number):02d}_*.csv"))
        pins[number] = sha256(path)
        rows = list(csv.DictReader(path.open()))
        actual[int(number)] = [
            [value for text in list(row.values())[skip:] for value in numbers(text)]
            for row in rows
        ]
    for number, count in target["protocol_rows"].items():
        path = next(table_dir.glob(f"table_{int(number):02d}_*.csv"))
        with path.open(newline="") as stream:
            if len(list(csv.DictReader(stream))) != count:
                raise ValueError(f"Protocol table row count differs: {number}")
        pins[number] = sha256(path)
    signature = (
        sha256(expected_path)
        + sha256(Path(__file__))
        + json.dumps(pins, sort_keys=True)
    )
    journal = Journal(Path(output_dir) / "checks.jsonl")
    for check in target["checks"]:
        key = ":".join(str(check[k]) for k in ["table", "row", "column"])
        old = journal.get(key, signature)
        if old is not None:
            if not old["passed"]:
                raise ValueError(f"Previously failed table check: {key}")
            continue
        value = actual[check["table"]][check["row"]][check["column"]]
        passed = math.isclose(value, check["expected"], rel_tol=0, abs_tol=1e-12)
        journal.save(key, signature, {**check, "actual": value, "passed": passed})
        if not passed:
            raise ValueError(f"Table check {key}: {value} != {check['expected']}")
    if sum(len(row) for rows in actual.values() for row in rows) != len(target["checks"]):
        raise ValueError("Numeric table coverage differs from expectations")
    receipt = dict(
        status="PASS",
        complete=True,
        empirical_tables=len(target["skip_columns"]),
        protocol_tables=len(target["protocol_rows"]),
        numeric_cells=len(target["checks"]),
        structured_cells=structured_cells,
        structured_expectations_sha256=sha256(cell_path),
        input_sha256=pins,
        expected_sha256=sha256(expected_path),
        paper_commit=target["paper_commit"],
    )
    write_json(Path(output_dir) / "summary.json", receipt)
    return receipt
