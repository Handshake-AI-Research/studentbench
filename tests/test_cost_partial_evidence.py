"""Independent regressions for partial charges and incomplete token accounting."""
import pytest
from studentbench.cost_reconstruction import reconstruct_session

RATE = dict(input_usd_per_million_tokens='2', output_usd_per_million_tokens='10', reasoning_tokens_additional='true')

def request(charge=''):
    return dict(student_id='synthetic-session', ai_model_preset_id='synthetic-arm', token_usage_recorded='true', prompt_tokens='1000', completion_tokens='100', reasoning_tokens='20', recorded_cost_usd=charge)

def test_partial_charges_without_replay_are_a_lower_bound():
    result = reconstruct_session([request('.006'), request()], RATE, '')
    assert result['replay_estimated_total_cost_usd'] is None
    assert result['best_available_total_cost_usd'] == .006
    assert result['best_available_cost_method'] == 'partial_evidence_lower_bound'

def test_missing_counts_on_one_metered_request_are_rejected():
    incomplete = request()
    incomplete.update(prompt_tokens='', completion_tokens='', reasoning_tokens='')
    with pytest.raises(ValueError):
        reconstruct_session([request(), incomplete], RATE, '.5')


@pytest.mark.parametrize('separate_reasoning, partial_count', [('true', 1), ('false', 0)])
def test_missing_reasoning_is_partial_only_when_billed_separately(separate_reasoning, partial_count):
    record = request()
    record['reasoning_tokens'] = ''
    rate = {**RATE, 'reasoning_tokens_additional': separate_reasoning}
    result = reconstruct_session([record], rate, '.5')
    assert result['partially_recorded_token_requests'] == partial_count
