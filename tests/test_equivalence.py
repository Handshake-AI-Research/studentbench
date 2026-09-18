"""The privacy-safe aggregate path must implement the same test as raw outcomes."""

import numpy as np
from scipy import stats

from studentbench.geography import tost, tost_from_moments


def test_moments_preserve_the_raw_welch_equivalence_test():
    ai = np.array([1, 3, 7, 5, 8, 4, 3, 9, 7, 2], dtype=float)
    human = np.array([2, 6, 8, 5, 4, 9, 8, 7], dtype=float)
    margin = 2.5
    actual = tost_from_moments(
        len(ai),
        ai.mean(),
        ai.var(ddof=1),
        len(human),
        human.mean(),
        human.var(ddof=1),
        margin,
    )
    lower = stats.ttest_ind(ai + margin, human, equal_var=False, alternative="greater")
    upper = stats.ttest_ind(ai - margin, human, equal_var=False, alternative="less")
    assert np.isclose(actual["p_lower"], lower.pvalue, atol=1e-14)
    assert np.isclose(actual["p_upper"], upper.pvalue, atol=1e-14)
    assert actual == tost(ai, human, margin)


def test_reversing_groups_preserves_equivalence_but_reverses_interval():
    ai = np.array([1, 4, 6, 9, 3, 5], dtype=float)
    human = np.array([2, 5, 8, 3, 7], dtype=float)
    forward, reverse = tost(ai, human, 3), tost(human, ai, 3)
    assert forward["equivalent"] == reverse["equivalent"]
    assert np.isclose(forward["p_tost"], reverse["p_tost"])
    assert np.isclose(forward["ci90_low_pp"], -reverse["ci90_high_pp"])
