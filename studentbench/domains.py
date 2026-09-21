"""Academic-domain comparisons for Figure 10 and Table 10.

Raw gains, Welch differences and baseline-adjusted ANCOVA answer distinct
questions; the same domain responses supply all three summaries.
"""

from __future__ import annotations
import math
from typing import Any, Iterable
import numpy as np
import pandas as pd
from scipy import stats
from .proficiency import DOMAIN_ORDER


def _mean_ci(
    values: Iterable[float], confidence: float
) -> tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        raise ValueError("Cannot estimate an empty group")
    mean = float(array.mean())
    if len(array) == 1:
        return mean, mean, mean, 1
    tail = (1.0 - confidence) / 2.0
    half = float(stats.t.ppf(1.0 - tail, len(array) - 1) * stats.sem(array))
    return mean, mean - half, mean + half, int(len(array))


def _bh_adjust(p_values: list[float]) -> list[float]:
    size = len(p_values)
    order = sorted(range(size), key=lambda index: p_values[index])
    adjusted = [1.0] * size
    running = 1.0
    for rank_index in range(size - 1, -1, -1):
        index = order[rank_index]
        rank = rank_index + 1
        running = min(running, p_values[index] * size / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def _welch_difference(
    treatment: np.ndarray, control: np.ndarray, confidence: float
) -> dict[str, float]:
    """Welch difference in participant-level gain, including its t interval."""
    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    estimate = float(treatment.mean() - control.mean())
    variance_t = float(treatment.var(ddof=1) / len(treatment))
    variance_c = float(control.var(ddof=1) / len(control))
    variance = variance_t + variance_c
    standard_error = math.sqrt(variance)
    degrees_freedom = variance**2 / (
        variance_t**2 / (len(treatment) - 1) + variance_c**2 / (len(control) - 1)
    )
    statistic = estimate / standard_error
    tail = (1.0 - confidence) / 2.0
    critical = float(stats.t.ppf(1.0 - tail, degrees_freedom))
    return {
        "estimate_pp": estimate,
        "ci_low_pp": estimate - critical * standard_error,
        "ci_high_pp": estimate + critical * standard_error,
        "se_welch": standard_error,
        "welch_df": float(degrees_freedom),
        "t": statistic,
        "p_two_sided": float(2.0 * stats.t.sf(abs(statistic), degrees_freedom)),
    }


def _hc3_domain_contrast(
    treatment: pd.DataFrame, control: pd.DataFrame, confidence: float
) -> dict[str, float | int]:
    """HC3 ANCOVA for post percentage ~ pre percentage + treatment."""
    block = pd.concat(
        [treatment.assign(is_treatment=1.0), control.assign(is_treatment=0.0)]
    )
    pre = 100.0 * block.pre_correct.to_numpy(float) / block.n_items.to_numpy(float)
    post = 100.0 * block.post_correct.to_numpy(float) / block.n_items.to_numpy(float)
    design = np.column_stack(
        [np.ones(len(block)), pre, block.is_treatment.to_numpy(float)]
    )
    inverse = np.linalg.inv(design.T @ design)
    beta = inverse @ design.T @ post
    residual = post - design @ beta
    leverage = np.sum((design @ inverse) * design, axis=1)
    if np.any(leverage >= 1.0):
        raise ValueError("Domain ANCOVA leverage is not below one")
    scaled = residual / (1.0 - leverage)
    meat = design.T @ ((scaled**2)[:, None] * design)
    covariance = inverse @ meat @ inverse
    estimate = float(beta[2])
    standard_error = float(math.sqrt(max(0.0, covariance[2, 2])))
    statistic = estimate / standard_error
    tail = (1.0 - confidence) / 2.0
    critical = float(stats.norm.ppf(1.0 - tail))
    return {
        "estimate_pp": estimate,
        "ci_low_pp": estimate - critical * standard_error,
        "ci_high_pp": estimate + critical * standard_error,
        "se_hc3": standard_error,
        "z": statistic,
        "p_two_sided_hc3": float(2.0 * stats.norm.sf(abs(statistic))),
        "n": int(len(block)),
    }


def _clustered_domain_profile_test(
    topics: pd.DataFrame, instrument: str, treatment_kind: str
) -> dict[str, Any]:
    """Test equality of topic-specific ANCOVA effects with student-clustered SEs."""
    domain_names = [
        topic for section, topic, _ in DOMAIN_ORDER if section == instrument
    ]
    block = topics[
        (topics.instrument == instrument)
        & topics.kind.isin([treatment_kind, "control"])
    ].copy()
    topic_index = {topic: index for index, topic in enumerate(domain_names)}
    domain_count = len(domain_names)
    design = np.zeros((len(block), 3 * domain_count), dtype=float)
    outcome = np.zeros(len(block), dtype=float)
    for row_number, row in enumerate(block.itertuples()):
        index = topic_index[row.topic]
        pre_pp = 100.0 * float(row.pre_correct) / float(row.n_items)
        post_pp = 100.0 * float(row.post_correct) / float(row.n_items)
        design[row_number, index] = 1.0
        design[row_number, domain_count + index] = pre_pp
        design[row_number, 2 * domain_count + index] = float(row.kind == treatment_kind)
        outcome[row_number] = post_pp
    inverse = np.linalg.inv(design.T @ design)
    beta = inverse @ design.T @ outcome
    residual = outcome - design @ beta
    meat = np.zeros_like(inverse)
    groups = block.student_id.astype(str).to_numpy()
    unique_groups = sorted(set(groups))
    for group in unique_groups:
        rows = groups == group
        score = design[rows].T @ residual[rows]
        meat += np.outer(score, score)
    observations, parameters = design.shape
    group_count = len(unique_groups)
    correction = (group_count / (group_count - 1.0)) * (
        (observations - 1.0) / (observations - parameters)
    )
    covariance = correction * inverse @ meat @ inverse
    effect_indexes = [2 * domain_count + index for index in range(domain_count)]
    effects = {
        domain: float(beta[index])
        for domain, index in zip(domain_names, effect_indexes)
    }

    restriction = np.zeros((domain_count - 1, parameters), dtype=float)
    for index in range(1, domain_count):
        restriction[index - 1, effect_indexes[0]] = 1.0
        restriction[index - 1, effect_indexes[index]] = -1.0
    difference = restriction @ beta
    restricted_covariance = restriction @ covariance @ restriction.T
    chi_square = float(difference @ np.linalg.pinv(restricted_covariance) @ difference)
    omnibus_p = float(stats.chi2.sf(chi_square, domain_count - 1))

    pairwise: list[dict[str, Any]] = []
    for left in range(domain_count):
        for right in range(left + 1, domain_count):
            contrast = np.zeros(parameters, dtype=float)
            contrast[effect_indexes[left]] = 1.0
            contrast[effect_indexes[right]] = -1.0
            estimate = float(contrast @ beta)
            standard_error = float(
                math.sqrt(max(0.0, contrast @ covariance @ contrast))
            )
            z_value = estimate / standard_error
            pairwise.append(
                {
                    "left_topic": domain_names[left],
                    "right_topic": domain_names[right],
                    "estimate_difference_pp": estimate,
                    "se_cluster": standard_error,
                    "z": z_value,
                    "p_two_sided_cluster": float(2.0 * stats.norm.sf(abs(z_value))),
                }
            )
    adjusted = _bh_adjust([float(row["p_two_sided_cluster"]) for row in pairwise])
    for row, value in zip(pairwise, adjusted):
        row["p_two_sided_cluster_bh_within_instrument"] = value
    return {
        "instrument": instrument,
        "contrast": f"{treatment_kind}_minus_control",
        "covariance": "student-clustered sandwich covariance with finite-sample CR1 correction",
        "n_rows": observations,
        "n_students": group_count,
        "topic_effects_pp": effects,
        "omnibus_equal_effects_chi_square": chi_square,
        "omnibus_equal_effects_df": domain_count - 1,
        "omnibus_equal_effects_p_two_sided": omnibus_p,
        "pairwise": pairwise,
    }


def build_effects(
    topic_rows: list[dict[str, Any]], confidence: float
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the final-p90 three-estimator domain comparison."""
    topics = pd.DataFrame(topic_rows)
    output: list[dict[str, Any]] = []
    for instrument, topic, n_items in DOMAIN_ORDER:
        block = topics[(topics.instrument == instrument) & (topics.topic == topic)]
        row: dict[str, Any] = {
            "instrument": instrument,
            "topic": topic,
            "n_items": n_items,
        }
        for kind, prefix in (("ai", "ai"), ("human", "human"), ("control", "control")):
            mean, low, high, n = _mean_ci(
                block.loc[block.kind == kind, "gain_pp"], confidence
            )
            row.update(
                {
                    f"raw_{prefix}_mean_pp": mean,
                    f"raw_{prefix}_ci_low_pp": low,
                    f"raw_{prefix}_ci_high_pp": high,
                    f"raw_{prefix}_n": n,
                }
            )
        control = block[block.kind == "control"]
        for kind in ("ai", "human"):
            treatment = block[block.kind == kind]
            welch = _welch_difference(
                treatment.gain_pp.to_numpy(float),
                control.gain_pp.to_numpy(float),
                confidence,
            )
            adjusted = _hc3_domain_contrast(treatment, control, confidence)
            for key, value in welch.items():
                row[f"welch_{kind}_minus_control_{key}"] = value
            for key, value in adjusted.items():
                row[f"ancova_{kind}_minus_control_{key}"] = value
        output.append(row)
    frame = pd.DataFrame(output)
    for instrument in ("quant", "verbal"):
        mask = frame.instrument == instrument
        for kind in ("ai", "human"):
            key = f"ancova_{kind}_minus_control_p_two_sided_hc3"
            adjusted = _bh_adjust(frame.loc[mask, key].astype(float).tolist())
            frame.loc[mask, f"{key}_bh_within_instrument"] = adjusted

    profile_tests: dict[str, Any] = {}
    maximum_consistency_delta = 0.0
    for instrument in ("quant", "verbal"):
        for kind in ("ai", "human"):
            result = _clustered_domain_profile_test(topics, instrument, kind)
            profile_tests[f"{instrument}_{kind}_minus_control"] = result
            for topic, stacked_effect in result["topic_effects_pp"].items():
                separate = float(
                    frame.loc[
                        (frame.instrument == instrument) & (frame.topic == topic),
                        f"ancova_{kind}_minus_control_estimate_pp",
                    ].iloc[0]
                )
                maximum_consistency_delta = max(
                    maximum_consistency_delta, abs(separate - float(stacked_effect))
                )
    profile_tests["validation"] = {
        "stacked_coefficients_match_separate_ancova": maximum_consistency_delta < 1e-9,
        "maximum_absolute_coefficient_difference_pp": maximum_consistency_delta,
    }
    if maximum_consistency_delta >= 1e-9:
        raise ValueError(
            f"Stacked and separate domain ANCOVA effects disagree: {maximum_consistency_delta}"
        )
    frame = frame.sort_values(
        ["ancova_ai_minus_control_estimate_pp", "topic"], ascending=[False, True]
    ).reset_index(drop=True)
    frame.insert(0, "rank_by_ai_ancova", np.arange(1, len(frame) + 1))
    return frame, profile_tests
