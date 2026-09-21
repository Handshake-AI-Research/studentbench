"""Synthetic mutations prove costs depend on measurements, not reference totals."""
from copy import deepcopy
import pytest
from studentbench.cost_reconstruction import reconstruct_session, verify_summary

RATE = dict(input_usd_per_million_tokens="2", output_usd_per_million_tokens="10",
            reasoning_tokens_additional="true")


def records():
    return [dict(student_id="s", ai_model_preset_id="m", token_usage_recorded="true",
                 prompt_tokens="1000", completion_tokens="100", reasoning_tokens="20",
                 recorded_cost_usd="")]


def test_replay_uses_tokens_reasoning_and_multiplier():
    result = reconstruct_session(records(), RATE, ".5")
    assert result["best_available_total_cost_usd"] == .0022
    assert result["recorded_cost_usd_sum"] is None
    changed = records(); changed[0]["prompt_tokens"] = "2000"
    mutated = reconstruct_session(changed, RATE, ".5")
    assert mutated["best_available_total_cost_usd"] == .0032
    expected = {k: "" if v is None else str(v) for k, v in result.items()}
    with pytest.raises(ValueError):
        verify_summary(mutated, expected)


def test_complete_charges_override_replay_and_preserve_zero():
    original = records(); original[0]["recorded_cost_usd"] = "0"
    assert reconstruct_session(original, RATE, ".5")["best_available_total_cost_usd"] == 0
    original[0]["recorded_cost_usd"] = ".006"
    assert reconstruct_session(original, RATE, ".5")["best_available_total_cost_usd"] == .006


def test_unmetered_calls_are_lower_bounds():
    original = records(); extra = deepcopy(original[0]); extra.update(token_usage_recorded="false",
        prompt_tokens="", completion_tokens="", reasoning_tokens="")
    assert reconstruct_session(original + [extra], RATE, ".5")["best_available_cost_method"] == "partial_evidence_lower_bound"


def test_missing_replay_price_fails():
    with pytest.raises(ValueError):
        reconstruct_session(records(), None, ".5")
