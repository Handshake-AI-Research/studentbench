"""Interrupted checkpoint writes must preserve completed work on restart."""

import pytest
from studentbench.journal import Journal


def test_restart_after_partial_write(tmp_path):
    path = tmp_path / "progress.jsonl"
    Journal(path).save("first", "input-sha", {"estimate": 1.25})
    with path.open("ab") as handle:
        handle.write(b'{"key":"unfinished')
    resumed = Journal(path)
    assert resumed.get("first", "input-sha") == {"estimate": 1.25}
    assert path.with_suffix(".jsonl.incomplete").read_bytes() == b'{"key":"unfinished'
    resumed.save("second", "input-sha", 2)
    assert Journal(path).get("second", "input-sha") == 2


def test_corruption_in_completed_record_is_not_ignored(tmp_path):
    path = tmp_path / "progress.jsonl"
    path.write_bytes(b"broken record\n")
    with pytest.raises(ValueError):
        Journal(path)


def test_restart_without_final_newline(tmp_path):
    path = tmp_path / "progress.jsonl"
    path.write_text('{"key":"first","signature":"sha","value":1}')
    Journal(path).save("second", "sha", 2)
    assert Journal(path).get("first", "sha") == 1
    assert Journal(path).get("second", "sha") == 2
