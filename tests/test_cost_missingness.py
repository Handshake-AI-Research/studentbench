"""Independent synthetic regressions for missing and mismatched cost evidence."""
from studentbench.cost_reconstruction import reconstruct_session

RATE = dict(input_usd_per_million_tokens='2', output_usd_per_million_tokens='10', reasoning_tokens_additional='true')

def row(metered=True, charge=''):
    return dict(student_id='s', ai_model_preset_id='m', token_usage_recorded=str(metered).lower(), prompt_tokens='1000' if metered else '', completion_tokens='100' if metered else '', reasoning_tokens='20' if metered else '', recorded_cost_usd=charge)

def test_missing_cost_is_not_zero():
    r = reconstruct_session([row()], RATE, '')
    assert r['best_available_total_cost_usd'] is None
    assert r['best_available_cost_method'] != 'replay_measured_session_estimate'

def test_wholly_unmetered_replay_stays_missing():
    r = reconstruct_session([row(False)], RATE, '.5')
    assert r['replay_estimated_total_cost_usd'] is None
    assert r['best_available_total_cost_usd'] is None

def test_charge_on_wrong_request_is_not_complete():
    r = reconstruct_session([row(), row(False, '.006')], RATE, '.5')
    assert r['best_available_total_cost_usd'] == r['replay_estimated_total_cost_usd']

def test_complete_recorded_cost_needs_no_replay_price():
    r = reconstruct_session([row(charge='.006')], None, '.5')
    assert r['best_available_total_cost_usd'] == .006
