"""Learning, proficiency and released-theta sensitivity calculations.

No plotting or report construction is mixed into these estimators.
"""

from __future__ import annotations
import csv
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from scipy import stats


QUESTION_COUNT = 27


SCOPES = ("quant", "verbal", "combined")


ARM_BAND_SPECS = (
    ("low", 4, 10, "15–37%"),
    ("middle", 11, 17, "41–63%"),
    ("high", 18, 24, "67–89%"),
)


DOMAIN_ORDER = (
    ("quant", "Data Analysis", 7),
    ("quant", "Geometry", 5),
    ("quant", "Arithmetic", 8),
    ("quant", "Algebra", 7),
    ("verbal", "Sentence Equivalence", 7),
    ("verbal", "Text Completion", 7),
    ("verbal", "Reading Comprehension", 13),
)


QUARTILE_IDS = ("Q1", "Q2", "Q3", "Q4")


MODEL_LABELS = {
    "gemini-3.1-pro-high": "Gemini 3.1 Pro (high)",
    "gemini-3.5-flash-low": "Gemini 3.5 Flash (low)",
    "gemini-3.6-flash-low": "Gemini 3.6 Flash (low)",
    "gemini-3.7-flash-medium": "Gemini 3.7 Flash (medium)",
    "gemma-4-31b-high": "Gemma 4 31B (high)",
    "gpt-5.4-mini-none": "GPT-5.4 mini (none)",
    "gpt-5.5-high": "GPT-5.5 (high)",
    "gpt-5.5-pro-med": "GPT-5.5 Pro (medium)",
    "kimi-k2.6": "Kimi K2.6",
    "opus-4.8-off": "Opus 4.8 (off)",
    "opus-4.8-xhigh": "Opus 4.8 (x-high)",
    "opus-5-high": "Opus 5 (high)",
    "sonnet-4.6-low": "Sonnet 4.6 (low)",
    "sonnet-5-low": "Sonnet 5 (low)",
    "human": "Human tutor",
    "control": "No-tutor control",
}


def _holm(values: Iterable[float]) -> list[float]:
    values = [float(value) for value in values]
    adjusted = [math.nan] * len(values)
    finite = sorted(
        (value, index) for index, value in enumerate(values) if math.isfinite(value)
    )
    running = 0.0
    total = len(finite)
    for rank, (value, index) in enumerate(finite):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted


def _eta_squared(groups: list[np.ndarray]) -> float:
    values = np.concatenate(groups)
    grand = float(values.mean())
    between = sum(len(group) * (float(group.mean()) - grand) ** 2 for group in groups)
    total = float(np.square(values - grand).sum())
    return float(between / total) if total else 0.0


def _family(arm_id: str) -> str:
    if arm_id == "human":
        return "Human"
    if arm_id == "control":
        return "Control"
    if arm_id.startswith(("gemini", "gemma")):
        return "Google"
    if arm_id.startswith(("gpt",)):
        return "OpenAI"
    if arm_id.startswith(("opus", "sonnet")):
        return "Anthropic"
    if arm_id.startswith("kimi"):
        return "Moonshot"
    raise ValueError(f"Unrecognized arm family: {arm_id}")


def _kind(arm_id: str) -> str:
    return arm_id if arm_id in {"human", "control"} else "ai"


def _normal_topic(section: str, value: str) -> str:
    topic = str(value).strip()
    if section == "verbal" and topic.startswith("Reading Comprehension"):
        return "Reading Comprehension"
    return topic


def _assessment_topics(path: Path, section: str) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"skill", "is_correct"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing topic fields {sorted(required)}")
        rows = list(reader)
    if len(rows) != QUESTION_COUNT:
        raise ValueError(
            f"{path}: expected {QUESTION_COUNT} assessment rows, found {len(rows)}"
        )
    for row in rows:
        topic = _normal_topic(section, row["skill"])
        counts.setdefault(topic, [0, 0])
        counts[topic][0] += int(str(row["is_correct"]).strip().lower() in {"1", "true"})
        counts[topic][1] += 1
    return {topic: (values[0], values[1]) for topic, values in counts.items()}


def _coverage(frame: pd.DataFrame) -> tuple[set[str], list[dict[str, Any]]]:
    quant = set(frame.loc[(frame.section == "quant") & (frame.kind == "ai"), "arm_id"])
    verbal = set(
        frame.loc[(frame.section == "verbal") & (frame.kind == "ai"), "arm_id"]
    )
    shared = quant & verbal
    if len(quant) != 13 or len(verbal) != 13 or len(shared) != 12:
        raise ValueError(
            f"Unexpected arm coverage: Q={len(quant)} V={len(verbal)} shared={len(shared)}"
        )
    if quant - shared != {"sonnet-4.6-low"} or verbal - shared != {"sonnet-5-low"}:
        raise ValueError(
            f"Unexpected section-specific arms: {quant - shared} / {verbal - shared}"
        )
    rows = []
    for arm_id in sorted(quant | verbal | {"human", "control"}):
        block = frame[frame.arm_id == arm_id]
        rows.append(
            {
                "arm_id": arm_id,
                "arm_label": MODEL_LABELS[arm_id],
                "kind": _kind(arm_id),
                "family": _family(arm_id),
                "n_quant": int((block.section == "quant").sum()),
                "n_verbal": int((block.section == "verbal").sum()),
                "included_in_combined_arm_analysis": bool(
                    arm_id in shared or arm_id in {"human", "control"}
                ),
                "coverage_note": (
                    "shared exact arm identity"
                    if arm_id in shared
                    else "section-specific arm; excluded from combined leaderboard"
                    if arm_id not in {"human", "control"}
                    else "reference condition fielded in both sections"
                ),
            }
        )
    return shared, rows


def _scope_block(frame: pd.DataFrame, scope: str, shared: set[str]) -> pd.DataFrame:
    if scope in {"quant", "verbal"}:
        return frame[frame.section == scope].copy()
    return frame[(frame.kind != "ai") | (frame.arm_id.isin(shared))].copy()


def _t_summary(values: np.ndarray) -> dict[str, float]:
    n = len(values)
    mean = float(values.mean())
    se = float(stats.sem(values))
    critical = float(stats.t.ppf(0.975, n - 1))
    t_value = mean / se
    return {
        "mean_gain_pp": mean,
        "se": se,
        "ci_low_pp": mean - critical * se,
        "ci_high_pp": mean + critical * se,
        "t_vs_zero": t_value,
        "df": n - 1,
        "p_gain_vs_zero_nominal": float(2.0 * stats.t.sf(abs(t_value), n - 1)),
    }


def _hc3(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise ValueError(
            f"ANCOVA design is rank deficient: rank={rank}, columns={x.shape[1]}"
        )
    inv = np.linalg.inv(x.T @ x)
    beta = inv @ x.T @ y
    residual = y - x @ beta
    leverage = np.sum((x @ inv) * x, axis=1)
    if np.any(leverage >= 1.0):
        raise ValueError("ANCOVA leverage is not below one")
    scaled = residual / (1.0 - leverage)
    covariance = inv @ (x.T @ ((scaled**2)[:, None] * x)) @ inv
    return {
        "beta": beta,
        "covariance": covariance,
        "rank": rank,
        "n": len(y),
        "residual_df": len(y) - x.shape[1],
        "max_leverage": float(leverage.max()),
    }


def _cluster_cr1(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    """OLS with a finite-sample CR1 sandwich for repeated topic rows."""
    rank = int(np.linalg.matrix_rank(x))
    if rank != x.shape[1]:
        raise ValueError(
            f"Clustered design is rank deficient: rank={rank}, columns={x.shape[1]}"
        )
    inverse = np.linalg.inv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    unique_groups = sorted(set(str(value) for value in groups))
    group_values = np.asarray([str(value) for value in groups], dtype=object)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for group in unique_groups:
        selected = group_values == group
        score = x[selected].T @ residual[selected]
        meat += np.outer(score, score)
    observations, parameters = x.shape
    group_count = len(unique_groups)
    if group_count <= 1 or observations <= parameters:
        raise ValueError(
            "Clustered covariance lacks residual or cluster degrees of freedom"
        )
    correction = (group_count / (group_count - 1.0)) * (
        (observations - 1.0) / (observations - parameters)
    )
    covariance = correction * inverse @ meat @ inverse
    return {
        "beta": beta,
        "covariance": covariance,
        "rank": rank,
        "n": observations,
        "clusters": group_count,
        "residual_df": observations - parameters,
        "cr1_correction": correction,
    }


def _wald(fit: dict[str, Any], contrast: np.ndarray) -> dict[str, Any]:
    """Wald test for one or more estimable linear restrictions."""
    matrix = np.atleast_2d(np.asarray(contrast, dtype=float))
    difference = matrix @ fit["beta"]
    covariance = matrix @ fit["covariance"] @ matrix.T
    covariance_rank = int(np.linalg.matrix_rank(covariance))
    if covariance_rank != matrix.shape[0]:
        raise ValueError(
            f"Restriction covariance is rank deficient: rank={covariance_rank}, rows={matrix.shape[0]}"
        )
    statistic = float(difference @ np.linalg.inv(covariance) @ difference)
    result: dict[str, Any] = {
        "wald_chi2": statistic,
        "df": int(matrix.shape[0]),
        "p_raw": float(stats.chi2.sf(statistic, matrix.shape[0])),
        "restriction_estimates": difference.tolist(),
    }
    if matrix.shape[0] == 1:
        estimate = float(difference[0])
        standard_error = float(math.sqrt(max(0.0, covariance[0, 0])))
        critical = float(stats.norm.ppf(0.975))
        result.update(
            {
                "estimate_pp": estimate,
                "se": standard_error,
                "ci_low_pp": estimate - critical * standard_error,
                "ci_high_pp": estimate + critical * standard_error,
                "z": estimate / standard_error,
            }
        )
    return result


def _linear(
    fit: dict[str, Any], vector: np.ndarray, subtract: float = 0.0
) -> dict[str, float]:
    estimate = float(vector @ fit["beta"] - subtract)
    se = float(math.sqrt(max(0.0, vector @ fit["covariance"] @ vector)))
    z = estimate / se
    critical = float(stats.norm.ppf(0.975))
    return {
        "adjusted_marginal_gain_pp": estimate,
        "se_hc3": se,
        "ci_low_pp": estimate - critical * se,
        "ci_high_pp": estimate + critical * se,
        "z_vs_zero": z,
        "p_gain_vs_zero_nominal_hc3": float(2.0 * stats.norm.sf(abs(z))),
    }


def _distribution_and_raw(
    frame: pd.DataFrame, shared: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    distribution_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for scope in SCOPES:
        block = _scope_block(frame, scope, shared)
        groups = [
            arm.gain_pp.to_numpy(float)
            for _, arm in block.loc[block.kind == "ai"].groupby("arm_id", sort=True)
        ]
        local = []
        for arm_id, arm in block.groupby("arm_id", sort=True):
            values = arm.gain_pp.to_numpy(float)
            summary = {
                "scope": scope,
                "arm_id": str(arm_id),
                "arm_label": MODEL_LABELS[str(arm_id)],
                "kind": _kind(str(arm_id)),
                "family": _family(str(arm_id)),
                "n": int(len(arm)),
                "n_quant": int((arm.section == "quant").sum()),
                "n_verbal": int((arm.section == "verbal").sum()),
                "median_gain_pp": float(np.median(values)),
                **_t_summary(values),
            }
            local.append(summary)
            counts = Counter(float(value) for value in values)
            for gain, count in sorted(counts.items()):
                distribution_rows.append(
                    {
                        "scope": scope,
                        "arm_id": str(arm_id),
                        "arm_label": MODEL_LABELS[str(arm_id)],
                        "kind": _kind(str(arm_id)),
                        "family": _family(str(arm_id)),
                        "gain_pp": gain,
                        "participant_count": int(count),
                        "arm_n": int(len(arm)),
                        "fraction_of_arm": float(count / len(arm)),
                    }
                )
        adjusted = _holm(row["p_gain_vs_zero_nominal"] for row in local)
        for row, p_holm in zip(local, adjusted):
            row["p_gain_vs_zero_holm_within_scope"] = p_holm
        ai_rows = sorted(
            (row for row in local if row["kind"] == "ai"),
            key=lambda row: (-row["mean_gain_pp"], row["arm_label"]),
        )
        reference_rows = sorted(
            (row for row in local if row["kind"] != "ai"),
            key=lambda row: {"human": 0, "control": 1}[row["kind"]],
        )
        for rank, row in enumerate(ai_rows, 1):
            row["raw_rank"] = rank
            row["display_order"] = rank
            row["rank_scope"] = "AI arms only"
        for offset, row in enumerate(reference_rows, len(ai_rows) + 1):
            row["raw_rank"] = None
            row["display_order"] = offset
            row["rank_scope"] = "unranked reference condition"
        local = ai_rows + reference_rows
        summary_rows.extend(local)
        welch = stats.f_oneway(*groups, equal_var=False)
        tests.append(
            {
                "scope": scope,
                "n_ai": int((block.kind == "ai").sum()),
                "n_ai_arms": int(block.loc[block.kind == "ai", "arm_id"].nunique()),
                "reference_conditions_displayed_but_excluded": ["human", "control"],
                "welch_f": float(welch.statistic),
                "welch_p_raw": float(welch.pvalue),
                "method": "Welch heteroskedastic one-way ANOVA across AI configurations only",
            }
        )
    adjusted_tests = _holm(row["welch_p_raw"] for row in tests)
    for row, p_holm in zip(tests, adjusted_tests):
        row["welch_p_holm_across_3_scopes"] = p_holm
    return pd.DataFrame(distribution_rows), pd.DataFrame(summary_rows), tests


def _adjusted(
    frame: pd.DataFrame, shared: set[str]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    tests: list[dict[str, Any]] = []
    for scope in SCOPES:
        block = _scope_block(frame, scope, shared).copy()
        arms = sorted(block.arm_id.unique())
        if "control" not in arms:
            raise ValueError(f"{scope}: no control reference")
        nonreference = [arm for arm in arms if arm != "control"]
        names = ["Intercept", "pre_pct", "pre_pct_sq"] + [
            f"arm:{arm}" for arm in nonreference
        ]
        if scope == "combined":
            names.append("section_verbal")
        matrix = []
        for row in block.itertuples(index=False):
            vector = [1.0, float(row.pre_pct), float(row.pre_pct) ** 2]
            vector.extend(float(row.arm_id == arm) for arm in nonreference)
            if scope == "combined":
                vector.append(float(row.section == "verbal"))
            matrix.append(vector)
        x = np.asarray(matrix, dtype=float)
        y = block.post_pct.to_numpy(float)
        fit = _hc3(x, y)
        mean_pre = float(block.pre_pct.mean())
        mean_pre_sq = float(np.mean(block.pre_pct.to_numpy(float) ** 2))
        mean_verbal = float((block.section == "verbal").mean())
        local = []
        vectors: dict[str, np.ndarray] = {}
        for arm_id in arms:
            vector = np.zeros(len(names), dtype=float)
            vector[names.index("Intercept")] = 1.0
            vector[names.index("pre_pct")] = mean_pre
            vector[names.index("pre_pct_sq")] = mean_pre_sq
            if arm_id != "control":
                vector[names.index(f"arm:{arm_id}")] = 1.0
            if scope == "combined":
                vector[names.index("section_verbal")] = mean_verbal
            vectors[str(arm_id)] = vector
            arm = block[block.arm_id == arm_id]
            local.append(
                {
                    "scope": scope,
                    "arm_id": str(arm_id),
                    "arm_label": MODEL_LABELS[str(arm_id)],
                    "kind": _kind(str(arm_id)),
                    "family": _family(str(arm_id)),
                    "n": int(len(arm)),
                    "n_quant": int((arm.section == "quant").sum()),
                    "n_verbal": int((arm.section == "verbal").sum()),
                    "standardization_population_n": int(len(block)),
                    **_linear(fit, vector, subtract=mean_pre),
                    "estimand": (
                        "average predicted post-test percentage if every displayed assignment received "
                        "this arm, standardized over the scope's empirical pre-test"
                        + (" and section" if scope == "combined" else "")
                        + " distribution, minus empirical mean pre-test percentage"
                    ),
                }
            )
        adjusted = _holm(row["p_gain_vs_zero_nominal_hc3"] for row in local)
        for row, p_holm in zip(local, adjusted):
            row["p_gain_vs_zero_holm_within_scope"] = p_holm
        ai_rows = sorted(
            (row for row in local if row["kind"] == "ai"),
            key=lambda row: (-row["adjusted_marginal_gain_pp"], row["arm_label"]),
        )
        reference_rows = sorted(
            (row for row in local if row["kind"] != "ai"),
            key=lambda row: {"human": 0, "control": 1}[row["kind"]],
        )
        for rank, row in enumerate(ai_rows, 1):
            row["adjusted_rank"] = rank
            row["display_order"] = rank
            row["rank_scope"] = "AI arms only"
        for offset, row in enumerate(reference_rows, len(ai_rows) + 1):
            row["adjusted_rank"] = None
            row["display_order"] = offset
            row["rank_scope"] = "unranked reference condition"
        local = ai_rows + reference_rows
        summary_rows.extend(local)
        ai_arms = sorted(block.loc[block.kind == "ai", "arm_id"].unique())
        ai_reference = ai_arms[0]
        contrast = np.zeros((len(ai_arms) - 1, len(names)), dtype=float)
        for row_index, arm_id in enumerate(ai_arms[1:]):
            contrast[row_index, names.index(f"arm:{arm_id}")] = 1.0
            contrast[row_index, names.index(f"arm:{ai_reference}")] = -1.0
        differences = contrast @ fit["beta"]
        contrast_covariance = contrast @ fit["covariance"] @ contrast.T
        wald = float(differences @ np.linalg.inv(contrast_covariance) @ differences)
        tests.append(
            {
                "scope": scope,
                "n": int(len(block)),
                "n_ai": int((block.kind == "ai").sum()),
                "n_ai_arms": len(ai_arms),
                "design_rank": fit["rank"],
                "residual_df": fit["residual_df"],
                "max_leverage": fit["max_leverage"],
                "formula": (
                    "post_pct ~ pre_pct + pre_pct^2 + arm"
                    + (" + section_verbal" if scope == "combined" else "")
                ),
                "reference_arm": "control",
                "ai_heterogeneity_reference": ai_reference,
                "human_and_control_excluded_from_ai_omnibus": True,
                "covariance": "HC3 heteroskedasticity-consistent",
                "ai_omnibus_wald_chi2": wald,
                "ai_omnibus_df": len(ai_arms) - 1,
                "ai_omnibus_p_raw": float(stats.chi2.sf(wald, len(ai_arms) - 1)),
            }
        )
    adjusted_tests = _holm(row["ai_omnibus_p_raw"] for row in tests)
    for row, p_holm in zip(tests, adjusted_tests):
        row["ai_omnibus_p_holm_across_3_scopes"] = p_holm
    return pd.DataFrame(summary_rows), tests


def _quartile_thresholds(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Mutually exclusive empirical quartiles with score ties kept together."""
    output: dict[str, dict[str, Any]] = {}
    for section in ("quant", "verbal"):
        block = frame[frame.section == section]
        values = block.pre_score_points.to_numpy(float)
        cutoffs = [
            float(np.quantile(values, q, method="linear")) for q in (0.25, 0.5, 0.75)
        ]
        if not all(value.is_integer() for value in cutoffs):
            raise ValueError(
                f"{section}: quartile cutoffs are not on the released score grid: {cutoffs}"
            )
        q25, q50, q75 = (int(value) for value in cutoffs)
        if not q25 < q50 < q75:
            raise ValueError(f"{section}: quartile cutoffs are not strictly ordered")
        specifications = (
            ("Q1", int(values.min()), q25, f"≤{q25}/27"),
            ("Q2", q25 + 1, q50, f"{q25 + 1}–{q50}/27"),
            ("Q3", q50 + 1, q75 - 1, f"{q50 + 1}–{q75 - 1}/27"),
            ("Q4", q75, int(values.max()), f"≥{q75}/27"),
        )
        rows: list[dict[str, Any]] = []
        for quartile, lower, upper, score_label in specifications:
            selected = block[
                (block.pre_score_points >= lower) & (block.pre_score_points <= upper)
            ]
            rows.append(
                {
                    "section": section,
                    "quartile": quartile,
                    "lower_score_points_inclusive": lower,
                    "upper_score_points_inclusive": upper,
                    "score_label": score_label,
                    "percentage_label": (
                        f"{100.0 * lower / QUESTION_COUNT:.0f}–{100.0 * upper / QUESTION_COUNT:.0f}%"
                        if lower != upper
                        else f"{100.0 * lower / QUESTION_COUNT:.0f}%"
                    ),
                    "n_all_conditions": int(len(selected)),
                    "fraction_all_conditions": float(len(selected) / len(block)),
                }
            )
        if sum(row["n_all_conditions"] for row in rows) != len(block):
            raise ValueError(
                f"{section}: quartile strata do not partition the physical cohort"
            )
        output[section] = {
            "full_physical_cohort_n": int(len(block)),
            "q25_points": q25,
            "q50_points": q50,
            "q75_points": q75,
            "q25_tied_n": int((block.pre_score_points == q25).sum()),
            "q50_tied_n": int((block.pre_score_points == q50).sum()),
            "q75_tied_n": int((block.pre_score_points == q75).sum()),
            "quartiles": rows,
            "rule": (
                "section-specific empirical 25th, 50th and 75th percentile score points; "
                "identical integer scores are never split; Q4 is inclusive at Q75"
            ),
        }
    return output


def _with_quartile(
    frame: pd.DataFrame, thresholds: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    output = frame.copy()
    quartiles: list[str] = []
    for row in output.itertuples(index=False):
        threshold = thresholds[str(row.section)]
        score = int(row.pre_score_points)
        if score <= threshold["q25_points"]:
            quartiles.append("Q1")
        elif score <= threshold["q50_points"]:
            quartiles.append("Q2")
        elif score < threshold["q75_points"]:
            quartiles.append("Q3")
        else:
            quartiles.append("Q4")
    output["proficiency_quartile"] = quartiles
    return output


def _welch_gain_difference(ai: np.ndarray, control: np.ndarray) -> dict[str, float]:
    ai = np.asarray(ai, dtype=float)
    control = np.asarray(control, dtype=float)
    variance_ai = float(ai.var(ddof=1) / len(ai))
    variance_control = float(control.var(ddof=1) / len(control))
    variance = variance_ai + variance_control
    standard_error = math.sqrt(variance)
    estimate = float(ai.mean() - control.mean())
    degrees_freedom = variance**2 / (
        variance_ai**2 / (len(ai) - 1) + variance_control**2 / (len(control) - 1)
    )
    statistic = estimate / standard_error
    critical = float(stats.t.ppf(0.975, degrees_freedom))
    return {
        "ai_minus_control_pp": estimate,
        "se": standard_error,
        "ci_low_pp": estimate - critical * standard_error,
        "ci_high_pp": estimate + critical * standard_error,
        "test_statistic": statistic,
        "df": float(degrees_freedom),
        "p_raw": float(2.0 * stats.t.sf(abs(statistic), degrees_freedom)),
    }


def _proficiency_pooled_ai_control(
    frame: pd.DataFrame,
    thresholds: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Absolute gains and pooled-AI minus control effects in four quartiles."""
    data = _with_quartile(frame, thresholds)
    data = data[data.kind.isin(["ai", "control"])].copy()
    group_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for section in ("quant", "verbal"):
        threshold_rows = {
            row["quartile"]: row for row in thresholds[section]["quartiles"]
        }
        for quartile in QUARTILE_IDS:
            block = data[
                (data.section == section) & (data.proficiency_quartile == quartile)
            ].copy()
            if set(block.kind) != {"ai", "control"}:
                raise ValueError(
                    f"{section}/{quartile}: pooled AI and control are both required"
                )
            meta = threshold_rows[quartile]
            raw_values: dict[str, np.ndarray] = {}
            for kind in ("ai", "control"):
                arm = block[block.kind == kind]
                values = arm.gain_pp.to_numpy(float)
                raw_values[kind] = values
                summary = _t_summary(values)
                group_rows.append(
                    {
                        "section": section,
                        "quartile": quartile,
                        "score_label": meta["score_label"],
                        "percentage_label": meta["percentage_label"],
                        "estimator": "raw absolute gain",
                        "condition": "pooled_ai" if kind == "ai" else "control",
                        "n": int(len(arm)),
                        "mean_gain_pp": summary["mean_gain_pp"],
                        "se": summary["se"],
                        "ci_low_pp": summary["ci_low_pp"],
                        "ci_high_pp": summary["ci_high_pp"],
                        "df": summary["df"],
                        "covariance": "participant-assignment t interval",
                        "estimand_type": "absolute pre-to-post gain, not a teaching-effect contrast",
                    }
                )
            raw_contrast = _welch_gain_difference(
                raw_values["ai"], raw_values["control"]
            )
            contrast_rows.append(
                {
                    "section": section,
                    "quartile": quartile,
                    "score_label": meta["score_label"],
                    "percentage_label": meta["percentage_label"],
                    "estimator": "raw Welch",
                    "n_ai": int((block.kind == "ai").sum()),
                    "n_control": int((block.kind == "control").sum()),
                    **raw_contrast,
                    "covariance": "Welch unequal-variance assignment-level",
                    "estimand_type": "pooled AI minus no-tutor teaching-effect contrast",
                }
            )

            block["ai_indicator"] = (block.kind == "ai").astype(float)
            pre = block.pre_pct.to_numpy(float)
            # A common linear term is used within all four narrow strata. Verbal Q3
            # contains only two released integer baseline scores (13 and 14), so a
            # within-stratum quadratic is not separately estimable there.
            x = np.column_stack(
                [np.ones(len(block)), pre, block.ai_indicator.to_numpy(float)]
            )
            fit = _hc3(x, block.post_pct.to_numpy(float))
            mean_pre = float(pre.mean())
            vectors = {
                "control": np.asarray([1.0, mean_pre, 0.0]),
                "pooled_ai": np.asarray([1.0, mean_pre, 1.0]),
            }
            for condition, vector in vectors.items():
                estimate = _linear(fit, vector, subtract=mean_pre)
                arm_kind = "ai" if condition == "pooled_ai" else "control"
                group_rows.append(
                    {
                        "section": section,
                        "quartile": quartile,
                        "score_label": meta["score_label"],
                        "percentage_label": meta["percentage_label"],
                        "estimator": "ANCOVA-adjusted absolute gain",
                        "condition": condition,
                        "n": int((block.kind == arm_kind).sum()),
                        "mean_gain_pp": estimate["adjusted_marginal_gain_pp"],
                        "se": estimate["se_hc3"],
                        "ci_low_pp": estimate["ci_low_pp"],
                        "ci_high_pp": estimate["ci_high_pp"],
                        "df": fit["residual_df"],
                        "covariance": "HC3 heteroskedasticity-consistent",
                        "estimand_type": (
                            "baseline-standardized absolute gain within the released quartile; "
                            "not a teaching-effect contrast"
                        ),
                    }
                )
            contrast = _wald(fit, np.asarray([[0.0, 0.0, 1.0]]))
            contrast_rows.append(
                {
                    "section": section,
                    "quartile": quartile,
                    "score_label": meta["score_label"],
                    "percentage_label": meta["percentage_label"],
                    "estimator": "ANCOVA HC3",
                    "n_ai": int((block.kind == "ai").sum()),
                    "n_control": int((block.kind == "control").sum()),
                    "ai_minus_control_pp": contrast["estimate_pp"],
                    "se": contrast["se"],
                    "ci_low_pp": contrast["ci_low_pp"],
                    "ci_high_pp": contrast["ci_high_pp"],
                    "test_statistic": contrast["z"],
                    "df": fit["residual_df"],
                    "p_raw": contrast["p_raw"],
                    "covariance": "HC3 heteroskedasticity-consistent at released assignment level",
                    "formula": "post_pct ~ pre_pct + pooled_ai within the released score-tied stratum",
                    "estimand_type": "pooled AI minus no-tutor teaching-effect contrast",
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    for estimator in ("raw Welch", "ANCOVA HC3"):
        selected = contrasts.estimator == estimator
        adjusted = _holm(contrasts.loc[selected, "p_raw"].astype(float).tolist())
        contrasts.loc[selected, "p_holm_across_8_quartile_contrasts"] = adjusted
    contrasts["significant_nominal_0_05"] = contrasts.p_raw < 0.05
    contrasts["significant_holm_0_05"] = (
        contrasts.p_holm_across_8_quartile_contrasts < 0.05
    )
    return pd.DataFrame(group_rows), contrasts


def _proficiency_pooled_ai_human(
    frame: pd.DataFrame,
    thresholds: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Direct pooled-AI minus human contrasts and proficiency interactions."""
    data = _with_quartile(frame, thresholds)
    data = data[data.kind.isin(["ai", "human"])].copy()
    contrast_rows: list[dict[str, Any]] = []
    for section in ("quant", "verbal"):
        threshold_rows = {
            row["quartile"]: row for row in thresholds[section]["quartiles"]
        }
        for quartile in QUARTILE_IDS:
            block = data[
                (data.section == section) & (data.proficiency_quartile == quartile)
            ].copy()
            if set(block.kind) != {"ai", "human"}:
                raise ValueError(
                    f"{section}/{quartile}: pooled AI and human tutoring are both required"
                )
            meta = threshold_rows[quartile]
            ai_gain = block.loc[block.kind == "ai", "gain_pp"].to_numpy(float)
            human_gain = block.loc[block.kind == "human", "gain_pp"].to_numpy(float)
            raw = _welch_gain_difference(ai_gain, human_gain)
            contrast_rows.append(
                {
                    "section": section,
                    "quartile": quartile,
                    "score_label": meta["score_label"],
                    "percentage_label": meta["percentage_label"],
                    "estimator": "raw Welch",
                    "n_ai": int(len(ai_gain)),
                    "n_human": int(len(human_gain)),
                    "ai_minus_human_pp": raw["ai_minus_control_pp"],
                    **{
                        key: value
                        for key, value in raw.items()
                        if key != "ai_minus_control_pp"
                    },
                    "covariance": "Welch unequal-variance assignment-level",
                    "formula": "raw participant gain in pooled AI minus human tutoring",
                    "estimand_type": "pooled AI minus live-human tutoring difference",
                }
            )

            block["ai_indicator"] = (block.kind == "ai").astype(float)
            pre = block.pre_pct.to_numpy(float)
            matrix = np.column_stack(
                [
                    np.ones(len(block)),
                    pre,
                    block.ai_indicator.to_numpy(float),
                ]
            )
            fit = _hc3(matrix, block.post_pct.to_numpy(float))
            adjusted = _wald(fit, np.asarray([0.0, 0.0, 1.0]))
            contrast_rows.append(
                {
                    "section": section,
                    "quartile": quartile,
                    "score_label": meta["score_label"],
                    "percentage_label": meta["percentage_label"],
                    "estimator": "ANCOVA HC3",
                    "n_ai": int((block.kind == "ai").sum()),
                    "n_human": int((block.kind == "human").sum()),
                    "ai_minus_human_pp": adjusted["estimate_pp"],
                    "se": adjusted["se"],
                    "ci_low_pp": adjusted["ci_low_pp"],
                    "ci_high_pp": adjusted["ci_high_pp"],
                    "test_statistic": adjusted["z"],
                    "df": fit["residual_df"],
                    "p_raw": adjusted["p_raw"],
                    "covariance": "HC3 heteroskedasticity-consistent at released assignment level",
                    "formula": "post_pct ~ pre_pct + pooled_ai within the released score-tied stratum; human reference",
                    "estimand_type": "pooled AI minus live-human tutoring difference",
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    for estimator in ("raw Welch", "ANCOVA HC3"):
        selected = contrasts.estimator == estimator
        contrasts.loc[selected, "p_holm_across_8_quartile_contrasts"] = _holm(
            contrasts.loc[selected, "p_raw"].astype(float).tolist()
        )
    contrasts["significant_nominal_0_05"] = contrasts.p_raw < 0.05
    contrasts["significant_holm_0_05"] = (
        contrasts.p_holm_across_8_quartile_contrasts < 0.05
    )

    data["ai"] = (data.kind == "ai").astype(float)
    data["verbal"] = (data.section == "verbal").astype(float)
    section_means = data.groupby("section").pre_pct.transform("mean")
    data["pre_centered"] = data.pre_pct - section_means
    quartile_columns = [
        (data.proficiency_quartile == quartile).to_numpy(float)
        for quartile in QUARTILE_IDS[1:]
    ]
    ai = data.ai.to_numpy(float)
    verbal = data.verbal.to_numpy(float)
    pre = data.pre_centered.to_numpy(float)
    categorical_names = [
        "Intercept",
        "pre_centered",
        "pre_centered_sq",
        "section_verbal",
        "section_verbal:pre_centered",
        "section_verbal:pre_centered_sq",
        "pooled_ai",
    ]
    categorical_names.extend(f"quartile:{q}" for q in QUARTILE_IDS[1:])
    categorical_names.extend(f"pooled_ai:quartile:{q}" for q in QUARTILE_IDS[1:])
    categorical_names.append("pooled_ai:section_verbal")
    categorical_names.extend(f"quartile:{q}:section_verbal" for q in QUARTILE_IDS[1:])
    categorical_names.extend(
        f"pooled_ai:quartile:{q}:section_verbal" for q in QUARTILE_IDS[1:]
    )
    categorical_matrix = [
        np.ones(len(data)),
        pre,
        pre**2,
        verbal,
        verbal * pre,
        verbal * pre**2,
        ai,
    ]
    categorical_matrix.extend(quartile_columns)
    categorical_matrix.extend(ai * column for column in quartile_columns)
    categorical_matrix.append(ai * verbal)
    categorical_matrix.extend(column * verbal for column in quartile_columns)
    categorical_matrix.extend(ai * column * verbal for column in quartile_columns)
    categorical_fit = _hc3(
        np.column_stack(categorical_matrix), data.post_pct.to_numpy(float)
    )

    tests: list[dict[str, Any]] = []
    for section in ("quant", "verbal"):
        restriction = np.zeros((3, len(categorical_names)), dtype=float)
        for index, quartile in enumerate(QUARTILE_IDS[1:]):
            restriction[
                index, categorical_names.index(f"pooled_ai:quartile:{quartile}")
            ] = 1.0
            if section == "verbal":
                restriction[
                    index,
                    categorical_names.index(
                        f"pooled_ai:quartile:{quartile}:section_verbal"
                    ),
                ] = 1.0
        tests.append(
            {
                "analysis": "categorical four-quartile primary",
                "test": f"{section}_pooled_ai_minus_human_by_quartile",
                "section": section,
                **_wald(categorical_fit, restriction),
                "formula": (
                    "post_pct ~ instrument-specific quadratic baseline + pooled_ai * quartile * "
                    "section_verbal; human-tutor reference"
                ),
                "interpretation": (
                    "joint test that the pooled-AI minus human-tutor difference is constant "
                    "across four score-tied proficiency strata"
                ),
            }
        )
    cross_restriction = np.zeros((3, len(categorical_names)), dtype=float)
    for index, quartile in enumerate(QUARTILE_IDS[1:]):
        cross_restriction[
            index,
            categorical_names.index(f"pooled_ai:quartile:{quartile}:section_verbal"),
        ] = 1.0
    tests.append(
        {
            "analysis": "categorical four-quartile primary",
            "test": "pooled_ai_minus_human_by_quartile_by_instrument",
            "section": "cross-instrument",
            **_wald(categorical_fit, cross_restriction),
            "formula": (
                "post_pct ~ instrument-specific quadratic baseline + pooled_ai * quartile * "
                "section_verbal; human-tutor reference"
            ),
            "interpretation": (
                "joint three-way test of whether proficiency modification of the AI-minus-human "
                "difference varies between Quant and Verbal"
            ),
        }
    )

    continuous_pre = data.pre_pct.to_numpy(float) - 50.0
    continuous_names = [
        "Intercept",
        "pre_centered",
        "pre_centered_sq",
        "section_verbal",
        "pooled_ai",
        "pooled_ai:section_verbal",
        "section_verbal:pre_centered",
        "section_verbal:pre_centered_sq",
        "pooled_ai:pre_centered",
        "pooled_ai:pre_centered_sq",
        "pooled_ai:section_verbal:pre_centered",
        "pooled_ai:section_verbal:pre_centered_sq",
    ]
    continuous_matrix = np.column_stack(
        [
            np.ones(len(data)),
            continuous_pre,
            continuous_pre**2,
            verbal,
            ai,
            ai * verbal,
            verbal * continuous_pre,
            verbal * continuous_pre**2,
            ai * continuous_pre,
            ai * continuous_pre**2,
            ai * verbal * continuous_pre,
            ai * verbal * continuous_pre**2,
        ]
    )
    continuous_fit = _hc3(continuous_matrix, data.post_pct.to_numpy(float))
    for section in ("quant", "verbal"):
        restriction = np.zeros((2, len(continuous_names)), dtype=float)
        restriction[0, continuous_names.index("pooled_ai:pre_centered")] = 1.0
        restriction[1, continuous_names.index("pooled_ai:pre_centered_sq")] = 1.0
        if section == "verbal":
            restriction[
                0, continuous_names.index("pooled_ai:section_verbal:pre_centered")
            ] = 1.0
            restriction[
                1, continuous_names.index("pooled_ai:section_verbal:pre_centered_sq")
            ] = 1.0
        tests.append(
            {
                "analysis": "continuous quadratic baseline sensitivity",
                "test": f"{section}_pooled_ai_minus_human_by_continuous_baseline",
                "section": section,
                **_wald(continuous_fit, restriction),
                "formula": (
                    "post_pct ~ pooled_ai * section_verbal * ((pre_pct-50) + "
                    "(pre_pct-50)^2); human-tutor reference"
                ),
                "interpretation": (
                    "joint linear-plus-quadratic baseline modification of the pooled-AI minus "
                    "human-tutor difference"
                ),
            }
        )
    continuous_cross = np.zeros((2, len(continuous_names)), dtype=float)
    continuous_cross[
        0, continuous_names.index("pooled_ai:section_verbal:pre_centered")
    ] = 1.0
    continuous_cross[
        1, continuous_names.index("pooled_ai:section_verbal:pre_centered_sq")
    ] = 1.0
    tests.append(
        {
            "analysis": "continuous quadratic baseline sensitivity",
            "test": "pooled_ai_minus_human_by_continuous_baseline_by_instrument",
            "section": "cross-instrument",
            **_wald(continuous_fit, continuous_cross),
            "formula": (
                "post_pct ~ pooled_ai * section_verbal * ((pre_pct-50) + "
                "(pre_pct-50)^2); human-tutor reference"
            ),
            "interpretation": (
                "joint three-way sensitivity test of whether the continuous AI-minus-human "
                "difference curve varies between Quant and Verbal"
            ),
        }
    )
    for analysis in (
        "categorical four-quartile primary",
        "continuous quadratic baseline sensitivity",
    ):
        indexes = [
            index
            for index, row in enumerate(tests)
            if row["analysis"] == analysis and row["section"] in {"quant", "verbal"}
        ]
        for index, value in zip(
            indexes,
            _holm(tests[index]["p_raw"] for index in indexes),
        ):
            tests[index]["p_holm_across_quant_verbal"] = value
    cross_indexes = [
        index for index, row in enumerate(tests) if row["section"] == "cross-instrument"
    ]
    for index, value in zip(
        cross_indexes,
        _holm(tests[index]["p_raw"] for index in cross_indexes),
    ):
        tests[index]["p_holm_across_two_interaction_specifications"] = value
    for row in tests:
        contributing = (
            data
            if row["section"] == "cross-instrument"
            else data[data.section == row["section"]]
        )
        row.update(
            {
                "model_n_assignments": int(len(data)),
                "contributing_n_assignments": int(len(contributing)),
                "n_ai": int((contributing.kind == "ai").sum()),
                "n_human": int((contributing.kind == "human").sum()),
                "n_quant": int((contributing.section == "quant").sum()),
                "n_verbal": int((contributing.section == "verbal").sum()),
                "design_rank": (
                    categorical_fit["rank"]
                    if row["analysis"] == "categorical four-quartile primary"
                    else continuous_fit["rank"]
                ),
                "design_columns": (
                    len(categorical_names)
                    if row["analysis"] == "categorical four-quartile primary"
                    else len(continuous_names)
                ),
                "covariance": "HC3 at released assignment/session level",
                "continuous_baseline_center_pp": (
                    50.0
                    if row["analysis"] == "continuous quadratic baseline sensitivity"
                    else None
                ),
                "multiplicity": (
                    "Quant and Verbal tests Holm-adjusted within model specification; the two "
                    "cross-instrument specification tests Holm-adjusted together"
                ),
                "canonical_cross_instrument_person_cluster_available": False,
                "cross_instrument_person_clustering_performed": False,
            }
        )
    return contrasts, tests


def _configuration_topic_proficiency_frontier(
    topics: pd.DataFrame,
    thresholds: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Omnibus-first model/configuration resolution across topic and quartile."""
    data = _with_quartile(topics, thresholds)
    global_tests: list[dict[str, Any]] = []
    section_fits: dict[str, dict[str, Any]] = {}
    for section in ("quant", "verbal"):
        block = data[(data.section == section) & (data.kind == "ai")].copy()
        arms = sorted(block.arm_id.unique())
        domain_names = [
            topic for item_section, topic, _ in DOMAIN_ORDER if item_section == section
        ]
        cell_names = [
            f"cell:{arm}:{topic}:{quartile}"
            for arm in arms
            for topic in domain_names
            for quartile in QUARTILE_IDS
        ]
        names = cell_names + [f"pre:{topic}" for topic in domain_names]
        matrix = np.zeros((len(block), len(names)), dtype=float)
        for row_number, row in enumerate(block.itertuples(index=False)):
            cell = f"cell:{row.arm_id}:{row.topic}:{row.proficiency_quartile}"
            matrix[row_number, names.index(cell)] = 1.0
            matrix[row_number, names.index(f"pre:{row.topic}")] = float(
                row.pre_topic_pct
            )
        fit = _cluster_cr1(
            matrix,
            block.post_topic_pct.to_numpy(float),
            block.student_id.astype(str).to_numpy(),
        )
        reference_arm = arms[0]
        reference_topic = domain_names[0]
        restrictions: list[np.ndarray] = []
        for arm in arms[1:]:
            for topic in domain_names[1:]:
                for quartile in QUARTILE_IDS[1:]:
                    vector = np.zeros(len(names), dtype=float)
                    for coefficient, cell in (
                        (1.0, f"cell:{arm}:{topic}:{quartile}"),
                        (-1.0, f"cell:{reference_arm}:{topic}:{quartile}"),
                        (-1.0, f"cell:{arm}:{reference_topic}:{quartile}"),
                        (1.0, f"cell:{reference_arm}:{reference_topic}:{quartile}"),
                        (-1.0, f"cell:{arm}:{topic}:Q1"),
                        (1.0, f"cell:{reference_arm}:{topic}:Q1"),
                        (1.0, f"cell:{arm}:{reference_topic}:Q1"),
                        (-1.0, f"cell:{reference_arm}:{reference_topic}:Q1"),
                    ):
                        vector[names.index(cell)] += coefficient
                    restrictions.append(vector)
        result = _wald(fit, np.vstack(restrictions))
        global_tests.append(
            {
                "section": section,
                "test": "ai_configuration_by_topic_by_quartile",
                **result,
                "n_topic_rows": int(len(block)),
                "n_assignment_clusters": int(block.student_id.nunique()),
                "n_ai_arms": len(arms),
                "reference_arm": reference_arm,
                "reference_topic": reference_topic,
                "covariance": "study-session clustered CR1 across repeated topic rows within instrument",
                "hierarchy": (
                    "global three-way omnibus; cell-specific configuration tests and pairwise contrasts "
                    "are inferentially opened only if this gate survives Holm across Quant and Verbal"
                ),
            }
        )
        section_fits[section] = {
            "fit": fit,
            "names": names,
            "arms": arms,
            "topics": domain_names,
        }
    adjusted_global = _holm(row["p_raw"] for row in global_tests)
    for row, value in zip(global_tests, adjusted_global):
        row["p_holm_across_quant_verbal_global_frontiers"] = value
        row["global_gate_open_0_05"] = bool(value < 0.05)

    cell_rows: list[dict[str, Any]] = []
    detailed_effect_rows: list[dict[str, Any]] = []
    cell_fit_cache: dict[
        tuple[str, str, str], tuple[dict[str, Any], list[str], list[str]]
    ] = {}
    for section in ("quant", "verbal"):
        arms = section_fits[section]["arms"]
        global_gate = next(row for row in global_tests if row["section"] == section)[
            "global_gate_open_0_05"
        ]
        for topic in section_fits[section]["topics"]:
            for quartile in QUARTILE_IDS:
                cell = data[
                    (data.section == section)
                    & (data.topic == topic)
                    & (data.proficiency_quartile == quartile)
                    & (data.kind.isin(["ai", "control"]))
                ].copy()
                observed_arms = sorted(cell.loc[cell.kind == "ai", "arm_id"].unique())
                missing = sorted(set(arms) - set(observed_arms))
                if missing:
                    cell_rows.append(
                        {
                            "section": section,
                            "topic": topic,
                            "quartile": quartile,
                            "estimable": False,
                            "missing_ai_arms": "|".join(missing),
                            "n": int(len(cell)),
                            "global_gate_open": global_gate,
                        }
                    )
                    continue
                names = ["Intercept", "pre_topic_pct"] + [f"arm:{arm}" for arm in arms]
                matrix = np.column_stack(
                    [
                        np.ones(len(cell)),
                        cell.pre_topic_pct.to_numpy(float),
                        *[(cell.arm_id == arm).to_numpy(float) for arm in arms],
                    ]
                )
                fit = _hc3(matrix, cell.post_topic_pct.to_numpy(float))
                contrast = np.zeros((len(arms) - 1, len(names)), dtype=float)
                for index, arm in enumerate(arms[1:]):
                    contrast[index, names.index(f"arm:{arm}")] = 1.0
                    contrast[index, names.index(f"arm:{arms[0]}")] = -1.0
                omnibus = _wald(fit, contrast)
                estimates = {
                    arm: float(fit["beta"][names.index(f"arm:{arm}")]) for arm in arms
                }
                control = cell[cell.kind == "control"]
                for arm in arms:
                    arm_block = cell[(cell.kind == "ai") & (cell.arm_id == arm)]
                    vector = np.zeros(len(names), dtype=float)
                    vector[names.index(f"arm:{arm}")] = 1.0
                    result = _wald(fit, vector)
                    detailed_effect_rows.append(
                        {
                            "section": section,
                            "topic": topic,
                            "quartile": quartile,
                            "arm_id": arm,
                            "arm_label": MODEL_LABELS[arm],
                            "family": _family(arm),
                            "n_ai": int(len(arm_block)),
                            "n_control": int(len(control)),
                            "raw_absolute_ai_gain_pp": float(
                                arm_block.gain_topic_pp.mean()
                            ),
                            "raw_absolute_control_gain_pp": float(
                                control.gain_topic_pp.mean()
                            ),
                            "ancova_ai_minus_control_pp": result["estimate_pp"],
                            "ancova_se_hc3": result["se"],
                            "ancova_ci_low_pp": result["ci_low_pp"],
                            "ancova_ci_high_pp": result["ci_high_pp"],
                            "ancova_z": result["z"],
                            "ancova_p_raw": result["p_raw"],
                            "small_cell_n_lt_10": bool(
                                len(arm_block) < 10 or len(control) < 10
                            ),
                            "formula": "post_topic_pct ~ pre_topic_pct + AI_configuration within topic and score-tied quartile",
                            "covariance": "HC3; one topic row per released assignment in this cell",
                            "estimand_type": "AI configuration minus no-tutor control topic teaching-effect contrast",
                        }
                    )
                leader = sorted(
                    arms, key=lambda arm: (-estimates[arm], MODEL_LABELS[arm])
                )[0]
                counts = cell[cell.kind == "ai"].groupby("arm_id").size()
                cell_rows.append(
                    {
                        "section": section,
                        "topic": topic,
                        "quartile": quartile,
                        "estimable": True,
                        "n": int(len(cell)),
                        "n_ai": int((cell.kind == "ai").sum()),
                        "n_control": int((cell.kind == "control").sum()),
                        "n_ai_arms": len(arms),
                        "minimum_ai_arm_n": int(counts.min()),
                        "maximum_ai_arm_n": int(counts.max()),
                        "ai_configuration_wald_chi2": omnibus["wald_chi2"],
                        "ai_configuration_df": omnibus["df"],
                        "ai_configuration_p_raw": omnibus["p_raw"],
                        "top_observed_ai_arm_id": leader,
                        "top_observed_ai_arm_label": MODEL_LABELS[leader],
                        "top_observed_ai_minus_control_adjusted_pp": estimates[leader],
                        "global_gate_open": global_gate,
                        "covariance": "HC3 within one topic-quartile assignment cell",
                    }
                )
                cell_fit_cache[(section, topic, quartile)] = (fit, names, arms)
    cells = pd.DataFrame(cell_rows)
    estimable = cells.estimable.astype(bool)
    cells.loc[estimable, "ai_configuration_p_holm_across_28_cells"] = _holm(
        cells.loc[estimable, "ai_configuration_p_raw"].astype(float).tolist()
    )
    cells["cell_gate_open"] = (
        cells.global_gate_open.astype(bool)
        & cells.estimable.astype(bool)
        & (cells.ai_configuration_p_holm_across_28_cells < 0.05)
    )
    cells["configuration_rank_released"] = cells.cell_gate_open
    cells["resolution_note"] = np.where(
        cells.cell_gate_open,
        "hierarchical global and cell omnibus gates opened; Holm pairwise comparisons computed",
        "no inferential model/configuration rank released for this cell",
    )

    pairwise_rows: list[dict[str, Any]] = []
    for cell in cells[cells.cell_gate_open].itertuples(index=False):
        fit, names, arms = cell_fit_cache[
            (str(cell.section), str(cell.topic), str(cell.quartile))
        ]
        estimates = {arm: float(fit["beta"][names.index(f"arm:{arm}")]) for arm in arms}
        ordered = sorted(arms, key=lambda arm: (-estimates[arm], MODEL_LABELS[arm]))
        local: list[dict[str, Any]] = []
        for left, right in combinations(ordered, 2):
            vector = np.zeros(len(names), dtype=float)
            vector[names.index(f"arm:{left}")] = 1.0
            vector[names.index(f"arm:{right}")] = -1.0
            result = _wald(fit, vector)
            local.append(
                {
                    "section": str(cell.section),
                    "topic": str(cell.topic),
                    "quartile": str(cell.quartile),
                    "left_arm_id": left,
                    "left_arm_label": MODEL_LABELS[left],
                    "right_arm_id": right,
                    "right_arm_label": MODEL_LABELS[right],
                    "left_minus_right_pp": result["estimate_pp"],
                    "se_hc3": result["se"],
                    "ci_low_pp": result["ci_low_pp"],
                    "ci_high_pp": result["ci_high_pp"],
                    "p_raw": result["p_raw"],
                    "leader_pair": bool(left == ordered[0]),
                }
            )
        adjusted = _holm(row["p_raw"] for row in local)
        for row, value in zip(local, adjusted):
            row["p_holm_all_ai_pairs_within_cell"] = value
            row["leader_difference_detected_after_holm"] = bool(
                row["leader_pair"] and value < 0.05
            )
        pairwise_rows.extend(local)
    detailed_effects = pd.DataFrame(detailed_effect_rows)
    if len(detailed_effects) != 364:
        raise ValueError(
            f"Expected 364 model×topic×quartile effects, found {len(detailed_effects)}"
        )
    detailed_effects["ancova_p_holm_across_364_model_topic_quartile_effects"] = _holm(
        detailed_effects.ancova_p_raw.astype(float).tolist()
    )
    detailed_effects["effect_significant_holm_0_05"] = (
        detailed_effects.ancova_p_holm_across_364_model_topic_quartile_effects < 0.05
    )
    return cells, global_tests, pd.DataFrame(pairwise_rows), detailed_effects


def _arm_pretest_bands(
    frame: pd.DataFrame,
    shared: set[str],
    adjusted: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize fixed, cross-arm pre-test bands without quantile rebinning."""
    if not np.allclose(frame.pre_score_points, np.round(frame.pre_score_points)):
        raise ValueError("Pre-test point scores are not integers")
    if int(frame.pre_score_points.min()) < 4 or int(frame.pre_score_points.max()) > 24:
        raise ValueError(
            "Observed pre-test points fall outside the prespecified 4–24 range"
        )

    rows: list[dict[str, Any]] = []
    for scope in SCOPES:
        block = _scope_block(frame, scope, shared).copy()
        rank_rows = adjusted[adjusted.scope == scope].sort_values("display_order")
        ordered_arms = rank_rows.arm_id.astype(str).tolist()
        rank_by_arm = {
            str(row.arm_id): (
                None if pd.isna(row.adjusted_rank) else int(row.adjusted_rank)
            )
            for row in rank_rows.itertuples(index=False)
        }
        display_by_arm = {
            str(row.arm_id): int(row.display_order)
            for row in rank_rows.itertuples(index=False)
        }
        group_specs: list[tuple[str, str, str, str, pd.DataFrame, int, int | None]] = [
            (
                "all_arms",
                "All arms marginal",
                "marginal",
                "All arms",
                block,
                0,
                None,
            )
        ]
        for arm_id in ordered_arms:
            group_specs.append(
                (
                    arm_id,
                    MODEL_LABELS[arm_id],
                    _kind(arm_id),
                    _family(arm_id),
                    block[block.arm_id == arm_id],
                    display_by_arm[arm_id],
                    rank_by_arm[arm_id],
                )
            )

        for (
            arm_id,
            label,
            kind,
            family,
            arm,
            display_order,
            adjusted_rank,
        ) in group_specs:
            row_type = "marginal" if arm_id == "all_arms" else "arm"
            for band_index, (band_id, low, high, percent_label) in enumerate(
                ARM_BAND_SPECS
            ):
                bucket = arm[
                    (arm.pre_score_points >= low) & (arm.pre_score_points <= high)
                ]
                if len(bucket) < 2:
                    raise ValueError(
                        f"{scope}/{arm_id}/{band_id}: insufficient observations"
                    )
                values = bucket.gain_pp.to_numpy(float)
                summary = _t_summary(values)
                rows.append(
                    {
                        "scope": scope,
                        "row_type": row_type,
                        "arm_id": arm_id,
                        "arm_label": label,
                        "kind": kind,
                        "family": family,
                        "adjusted_rank": adjusted_rank,
                        "display_order": display_order,
                        "band_index": band_index,
                        "band_id": band_id,
                        "pretest_band_pct": percent_label,
                        "pretest_points_low_inclusive": low,
                        "pretest_points_high_inclusive": high,
                        "n": int(len(bucket)),
                        "n_quant": int((bucket.section == "quant").sum()),
                        "n_verbal": int((bucket.section == "verbal").sum()),
                        "mean_pretest_pp": float(bucket.pre_pct.mean()),
                        "mean_gain_pp": summary["mean_gain_pp"],
                        "se_pp": summary["se"],
                        "ci_low_pp": summary["ci_low_pp"],
                        "ci_high_pp": summary["ci_high_pp"],
                        "t_df": int(summary["df"]),
                        "small_cell_n_lt_10": bool(len(bucket) < 10),
                        "estimand": (
                            "raw assignment-level mean within-student gain in a fixed pre-test band; "
                            "descriptive, not an arm-by-baseline causal effect"
                        ),
                    }
                )

        scope_rows = [row for row in rows if row["scope"] == scope]
        expected_arm_rows = len(ordered_arms) * len(ARM_BAND_SPECS)
        observed_arm_rows = sum(row["row_type"] == "arm" for row in scope_rows)
        if observed_arm_rows != expected_arm_rows:
            raise ValueError(f"{scope}: incomplete arm-by-band grid")
        marginal_n = sum(
            row["n"] for row in scope_rows if row["row_type"] == "marginal"
        )
        if marginal_n != len(block):
            raise ValueError(
                f"{scope}: marginal fixed-band counts do not partition scope"
            )
        arm_n = sum(row["n"] for row in scope_rows if row["row_type"] == "arm")
        if arm_n != len(block):
            raise ValueError(
                f"{scope}: named-arm fixed-band counts do not partition scope"
            )

    return pd.DataFrame(rows).reset_index(drop=True)
