"""Verify every printed numeric cell in the paper's twelve empirical tables."""

from pathlib import Path
import csv
import json
import math
import re
from .data import sha256
from .journal import Journal, write_json


def numbers(value):
    text = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value).replace("−", "-")
    return [float(x) for x in re.findall(r"(?<![A-Za-z])[-+]?(?:\d*\.\d+|\d+)", text)]


def run(table_dir, output_dir, expected_path=None):
    table_dir = Path(table_dir)
    expected_path = (
        Path(expected_path)
        if expected_path
        else Path(__file__).resolve().parents[1]
        / "verification/expected_table_numbers.json"
    )
    target = json.loads(expected_path.read_text())
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
        assert len(list(csv.DictReader(path.open()))) == count
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
            assert old["passed"], key
            continue
        value = actual[check["table"]][check["row"]][check["column"]]
        passed = math.isclose(value, check["expected"], rel_tol=0, abs_tol=1e-12)
        journal.save(key, signature, {**check, "actual": value, "passed": passed})
        assert passed, (key, value, check["expected"])
    assert sum(len(row) for rows in actual.values() for row in rows) == len(
        target["checks"]
    )
    receipt = dict(
        status="PASS",
        complete=True,
        empirical_tables=12,
        protocol_tables=4,
        numeric_cells=len(target["checks"]),
        input_sha256=pins,
        expected_sha256=sha256(expected_path),
        paper_commit=target["paper_commit"],
    )
    write_json(Path(output_dir) / "summary.json", receipt)
    return receipt
