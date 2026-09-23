"""Compare the aggregate CR2 path to the direct residual-influence equations."""

import numpy as np
import pandas as pd
import pytest

from studentbench.primary_equivalence import cr2_moments, fit, session_fingerprint


def direct_fit(x, y, groups, contrast):
    bread = np.linalg.inv(x.T @ x)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    columns, scores = [], []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        block = x[indices]
        values, vectors = np.linalg.eigh(np.eye(len(indices)) - block @ bread @ block.T)
        u = (vectors / np.sqrt(values)) @ vectors.T @ block @ bread @ contrast
        influence = -x @ bread @ block.T @ u
        influence[indices] += u
        columns.append(influence)
        scores.append(u @ residual[indices])
    p = np.column_stack(columns)
    gram = p.T @ p
    return contrast @ beta, np.linalg.norm(scores), np.trace(gram)**2 / np.sum(gram**2)


@pytest.mark.parametrize("seed", [3, 15, 42])
def test_global_sums_match_direct_cr2_with_unequal_clusters(seed):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(8), [2, 5, 1, 7, 4, 3, 2, 1])
    x = np.column_stack([np.ones(len(groups)), rng.normal(size=(len(groups), 3))])
    y = x @ np.array([50., 4., 1., -3.]) + rng.normal(size=len(groups))
    contrast = np.array([0., .5, -.5, 0.])
    moments = cr2_moments(x, y, groups, contrast)
    actual = fit(x, y, contrast, moments)
    expected = direct_fit(x, y, groups, contrast)
    np.testing.assert_allclose([actual["estimate_pp"], actual["se_pp"], actual["df"]],
                               expected, rtol=1e-9, atol=1e-9)


def test_response_change_does_not_reuse_the_same_moment_fingerprint():
    frame = pd.DataFrame([dict(student_id="session1", section="quant", kind="ai",
                               pre_pct=30., post_pct=50., form_order="PQ")])
    fingerprint = session_fingerprint(frame)
    frame.loc[0, "post_pct"] = 60.
    assert session_fingerprint(frame) != fingerprint


def test_rank_deficiency_stops_inference():
    x = np.ones((6, 2))
    with pytest.raises(ValueError, match="full rank"):
        cr2_moments(x, np.arange(6.), np.arange(6), np.array([0., 1.]))
