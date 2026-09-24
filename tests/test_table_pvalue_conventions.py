"""The protocol table must not preserve a stale empirical passing count."""
import json

import pytest

from tables.table_05_pvalue_conventions import run


@pytest.mark.parametrize("passing, word", [(6, "six"), (5, "five"), (0, "zero")])
def test_table_passing_count_follows_computed_tests(tmp_path, passing, word):
    analysis = tmp_path / "analysis"
    costs = analysis / "costs"
    costs.mkdir(parents=True)
    with (costs / "individual_model_equivalence.jsonl").open("w") as stream:
        stream.write(json.dumps({"record_kind": "protocol"}) + "\n")
        for index in range(7):
            stream.write(json.dumps({
                "record_kind": "model_result", "nominal_equivalent": index < passing,
            }) + "\n")
    output = tmp_path / "tables"
    rows = run(tmp_path, analysis, output)
    expected = f"all {word} passing tutors"
    assert expected in rows[0]["Reporting rule"]
    assert expected in (output / "table_05_pvalue_conventions.csv").read_text()
    assert expected in (output / "table_05_pvalue_conventions_paper.tex").read_text()


def test_table_rejects_missing_individual_tests(tmp_path):
    costs = tmp_path / "costs"
    costs.mkdir()
    (costs / "individual_model_equivalence.jsonl").write_text(
        json.dumps({"record_kind": "protocol"}) + "\n"
    )
    with pytest.raises(ValueError, match="equivalence results are empty"):
        run(tmp_path, tmp_path, tmp_path / "tables")
