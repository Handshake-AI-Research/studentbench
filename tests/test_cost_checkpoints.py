"""Cost checkpoints must survive interruption and invalidate changed producers."""

import json

import pandas as pd
import pytest

from studentbench import costs


def test_producer_signature_covers_local_helpers(tmp_path, monkeypatch):
    for name in ("costs.py", "cost_reconstruction.py", "data.py", "journal.py"):
        (tmp_path / name).write_text("original")
    monkeypatch.setattr(costs, "__file__", str(tmp_path / "costs.py"))
    before = costs._producer_signature()
    for name in ("costs.py", "cost_reconstruction.py", "data.py", "journal.py"):
        (tmp_path / name).write_text("changed")
        assert costs._producer_signature() != before
        (tmp_path / name).write_text("original")
        assert costs._producer_signature() == before


def test_session_checkpoint_invalidates_changed_producer(tmp_path, monkeypatch):
    source = tmp_path / "data"
    student = (
        source / "1_main_leaderboard_data/gre_quant/control_arm/task_id_1/student_data"
    )
    student.mkdir(parents=True)
    (student / "student.json").write_text(
        json.dumps(
            {"student_id": "1", "pretest_score_points": 0, "posttest_score_points": 27}
        )
    )
    skills = [
        topic
        for section, topic, count in costs.DOMAIN_ORDER
        if section == "quant"
        for _ in range(count)
    ]
    for name, correct in (("pretest", 0), ("posttest", 1)):
        pd.DataFrame({"skill": skills, "is_correct": correct}).to_csv(
            student / f"{name}_responses.csv", index=False
        )
    monkeypatch.setattr(costs, "EXPECTED_COUNTS", {("quant", "control"): 1})
    producer = ["version-one"]
    monkeypatch.setattr(costs, "_producer_signature", lambda: producer[0])
    output = tmp_path / "results"
    original, _, input_hash, hits = costs._extract_students(source, output, force=False)
    assert hits == 0
    assert costs._extract_students(source, output, force=False)[3] == 1

    producer[0] = "version-two"
    refreshed, _, refreshed_input_hash, hits = costs._extract_students(
        source, output, force=False
    )
    assert hits == 0
    assert refreshed_input_hash == input_hash
    pd.testing.assert_frame_equal(original, refreshed)
    assert costs._extract_students(source, output, force=False)[3] == 1


def test_bootstrap_invalidates_producer_without_changing_seed(tmp_path, monkeypatch):
    block = pd.DataFrame(
        {
            "student_id": ["a", "b", "c", "d"],
            "gain_pp": [5.0, 10.0, 15.0, 20.0],
            "best_available_total_cost_usd": [0.02, 0.05, 0.07, 0.08],
        }
    )
    producer = ["version-one"]
    monkeypatch.setattr(costs, "_producer_signature", lambda: producer[0])
    path, cache = tmp_path / "bootstrap.jsonl", {}
    arguments = {
        "draws": 100,
        "seed_root": 20260912,
        "minimum_valid_fraction": 0.99,
        "confidence": 0.95,
    }
    original = costs._bootstrap_ratio(
        "quant", "test-model", block, path, cache, **arguments
    )
    resumed = costs._bootstrap_ratio(
        "quant", "test-model", block, path, cache, **arguments
    )
    assert resumed is original

    producer[0] = "version-two"
    refreshed = costs._bootstrap_ratio(
        "quant", "test-model", block, path, cache, **arguments
    )
    assert refreshed is not original
    assert refreshed["producer_signature"] != original["producer_signature"]
    assert {
        key: value for key, value in original.items() if key != "producer_signature"
    } == {key: value for key, value in refreshed.items() if key != "producer_signature"}
    assert len(path.read_text().splitlines()) == 2


def test_cost_journal_recovers_partial_final_record(tmp_path):
    path = tmp_path / "costs.jsonl"
    costs._append_jsonl(path, {"student_id": "first", "gain": 10})
    with path.open("ab") as handle:
        handle.write(b'{"student_id":"unfinished')
    assert set(costs._load_jsonl(path, "student_id")) == {"first"}
    assert (
        path.with_suffix(".jsonl.incomplete").read_bytes()
        == b'{"student_id":"unfinished'
    )
    costs._append_jsonl(path, {"student_id": "second", "gain": 20})
    assert set(costs._load_jsonl(path, "student_id")) == {"first", "second"}


def test_cost_journal_rejects_corrupt_completed_record(tmp_path):
    path = tmp_path / "costs.jsonl"
    path.write_bytes(b'broken\n{"student_id":"last"}\n')
    with pytest.raises(ValueError):
        costs._load_jsonl(path, "student_id")


def test_cost_journal_preserves_final_record_without_newline(tmp_path):
    path = tmp_path / "costs.jsonl"
    path.write_text('{"student_id":"first"}')
    costs._load_jsonl(path, "student_id")
    costs._append_jsonl(path, {"student_id": "second"})
    assert set(costs._load_jsonl(path, "student_id")) == {"first", "second"}
