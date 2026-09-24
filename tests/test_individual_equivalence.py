"""Checks of common-model subset fitting, fixed margins and input integrity."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from studentbench import primary_equivalence as primary
from studentbench.individual_equivalence import fit_model


def sample():
    rng = np.random.default_rng(71)
    n = 80
    frame = pd.DataFrame(dict(student_id=[f's{i}' for i in range(n)],
        section=np.repeat(['quant', 'verbal'], n // 2),
        kind=np.tile(np.repeat(['ai', 'human'], n // 4), 2),
        pre_pct=rng.uniform(10, 80, n), form_order=np.tile(['PQ', 'QP'], n // 2)))
    frame['arm_id'] = np.where(frame.kind == 'human', 'human', 'ai1')
    frame['human_tutor_id'] = np.where(frame.kind == 'human', 'tutor1', '')
    frame['post_pct'] = .5 * frame.pre_pct + rng.normal(30, 4, n)
    weights = dict(quant=.53, verbal=.47)
    x, c = primary.design(frame, 'combined', weights)
    source = dict(session_fingerprint=primary.session_fingerprint(frame),
                  moments=primary.cr2_moments(x, frame.post_pct, np.arange(n) // 2, c))
    return frame, source, weights


def test_individual_model_uses_primary_design_and_fixed_supplied_margin():
    frame, source, weights = sample()
    result = fit_model(frame, source, weights, 4.085)
    x, c = primary.design(frame, 'combined', weights)
    expected = primary.fit(x, frame.post_pct, c, source['moments'])
    assert result['estimate_pp'] == expected['estimate_pp']
    assert result['margin_pp'] == 4.085
    assert result['n_ai'] == result['n_human'] == 40
    p = max(stats.t.sf((result['estimate_pp'] + 4.085) / result['se_pp'], result['df']),
            stats.t.cdf((result['estimate_pp'] - 4.085) / result['se_pp'], result['df']))
    assert result['p_tost'] == p
    assert result['equivalent_at_05'] == (p < .05)


def test_changed_outcome_rejects_moments_before_fitting():
    frame, source, weights = sample()
    frame.loc[0, 'post_pct'] += 1
    with pytest.raises(ValueError, match='observations differ'):
        fit_model(frame, source, weights, 4.085)


def test_changed_section_weights_change_adjusted_contrast():
    frame, source, weights = sample()
    original = fit_model(frame, source, weights, 4.085)
    changed = dict(quant=.8, verbal=.2)
    x, c = primary.design(frame, 'combined', changed)
    source['moments'] = primary.cr2_moments(x, frame.post_pct, np.arange(len(frame)) // 2, c)
    assert fit_model(frame, source, changed, 4.085)['estimate_pp'] != original['estimate_pp']
