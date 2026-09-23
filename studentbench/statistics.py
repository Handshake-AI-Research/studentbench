"""Primary learning estimators: HC3 ANCOVA, Welch contrasts and TOST."""

from __future__ import annotations
import math
from typing import Any
import numpy as np
from scipy import stats


KINDS = ("ai", "human", "control")


SCOPES = ("quant", "verbal", "combined")


KIND_LABELS = {
    "ai": "AI tutors (pooled)",
    "human": "Human tutor",
    "control": "Control (no tutor)",
}


def _hc3_fit(block: list[dict[str, Any]], combined: bool) -> dict[str, Any]:
    names = ["Intercept", "pre_pct", "pre_pct_sq", "is_ai", "is_human"]
    if combined:
        names.append("section_verbal")
    matrix = []
    y = []
    for row in block:
        vector = [
            1.0,
            float(row["pre_pct"]),
            float(row["pre_pct"]) ** 2,
            float(row["kind"] == "ai"),
            float(row["kind"] == "human"),
        ]
        if combined:
            vector.append(float(row["section"] == "verbal"))
        matrix.append(vector)
        y.append(float(row["post_pct"]))
    x = np.asarray(matrix, dtype=float)
    outcome = np.asarray(y, dtype=float)
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ outcome
    residual = outcome - x @ beta
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    if np.any(leverage >= 1.0):
        raise ValueError("ANCOVA leverage is not below one")
    scaled = residual / (1.0 - leverage)
    meat = x.T @ ((scaled**2)[:, None] * x)
    covariance = xtx_inv @ meat @ xtx_inv
    return {
        "names": names,
        "x": x,
        "y": outcome,
        "beta": beta,
        "covariance": covariance,
        "rank": int(np.linalg.matrix_rank(x)),
        "n": len(block),
        "residual_df": len(block) - x.shape[1],
    }


def _linear_estimate(
    fit: dict[str, Any], vector: np.ndarray, *, subtract_constant: float = 0.0
) -> dict[str, float]:
    estimate = float(vector @ fit["beta"] - subtract_constant)
    se = float(math.sqrt(max(0.0, vector @ fit["covariance"] @ vector)))
    z_value = estimate / se
    critical = float(stats.norm.ppf(0.975))
    return {
        "estimate_pp": estimate,
        "se_hc3": se,
        "ci_low_pp": estimate - critical * se,
        "ci_high_pp": estimate + critical * se,
        "z": z_value,
        "p_two_sided_hc3": float(2.0 * stats.norm.sf(abs(z_value))),
    }


def _estimate_outcomes(records: list[dict[str, Any]]) -> dict[str, Any]:
    raw_groups: list[dict[str, Any]] = []
    adjusted_groups: list[dict[str, Any]] = []
    adjusted_contrasts: list[dict[str, Any]] = []
    models: list[dict[str, Any]] = []

    for scope in SCOPES:
        block = [
            row for row in records if scope == "combined" or row["section"] == scope
        ]
        arrays = {
            kind: np.asarray(
                [row["gain_pp"] for row in block if row["kind"] == kind], dtype=float
            )
            for kind in KINDS
        }
        for kind, values in arrays.items():
            n = len(values)
            mean = float(values.mean())
            se = float(stats.sem(values))
            critical = float(stats.t.ppf(0.975, n - 1))
            t_value = mean / se
            raw_groups.append(
                {
                    "scope": scope,
                    "kind": kind,
                    "label": KIND_LABELS[kind],
                    "n": n,
                    "mean_gain_pp": mean,
                    "se": se,
                    "ci_low_pp": mean - critical * se,
                    "ci_high_pp": mean + critical * se,
                    "t_vs_zero": t_value,
                    "df": n - 1,
                    "p_two_sided_gain_vs_zero": float(
                        2.0 * stats.t.sf(abs(t_value), n - 1)
                    ),
                    "ci_method": "participant-level Student-t 95% CI",
                }
            )

        fit = _hc3_fit(block, combined=scope == "combined")
        names = fit["names"]
        mean_pre = float(np.mean([row["pre_pct"] for row in block]))
        mean_pre_sq = float(np.mean([float(row["pre_pct"]) ** 2 for row in block]))
        mean_verbal = float(np.mean([row["section"] == "verbal" for row in block]))

        marginal_vectors: dict[str, np.ndarray] = {}
        for kind in KINDS:
            vector = np.zeros(len(names), dtype=float)
            vector[names.index("Intercept")] = 1.0
            vector[names.index("pre_pct")] = mean_pre
            vector[names.index("pre_pct_sq")] = mean_pre_sq
            vector[names.index("is_ai")] = float(kind == "ai")
            vector[names.index("is_human")] = float(kind == "human")
            if scope == "combined":
                vector[names.index("section_verbal")] = mean_verbal
            marginal_vectors[kind] = vector
            adjusted_groups.append(
                {
                    "scope": scope,
                    "kind": kind,
                    "label": KIND_LABELS[kind],
                    "n": len(arrays[kind]),
                    "standardization_population_n": len(block),
                    "adjusted_marginal_gain_pp": None,
                    **_linear_estimate(fit, vector, subtract_constant=mean_pre),
                    "estimand": (
                        "average predicted post-test percentage under the group assignment, "
                        "standardized over the scope's empirical pre-test and section distribution, "
                        "minus the scope's empirical mean pre-test percentage"
                    ),
                }
            )
            adjusted_groups[-1]["adjusted_marginal_gain_pp"] = adjusted_groups[-1][
                "estimate_pp"
            ]

        adjusted_pairs = (
            ("pooled_ai_minus_control", "ai", "control"),
            ("human_minus_control", "human", "control"),
            ("human_minus_pooled_ai", "human", "ai"),
        )
        for contrast, left, right in adjusted_pairs:
            vector = marginal_vectors[left] - marginal_vectors[right]
            adjusted_contrasts.append(
                {
                    "scope": scope,
                    "contrast": contrast,
                    "left_kind": left,
                    "right_kind": right,
                    **_linear_estimate(fit, vector),
                    "model": (
                        "post_pct ~ pre_pct + pre_pct^2 + group"
                        + (" + section fixed effect" if scope == "combined" else "")
                        + "; HC3 covariance"
                    ),
                }
            )

        indices = [names.index("is_ai"), names.index("is_human")]
        group_beta = fit["beta"][indices]
        group_cov = fit["covariance"][np.ix_(indices, indices)]
        wald = float(group_beta @ np.linalg.inv(group_cov) @ group_beta)
        models.append(
            {
                "scope": scope,
                "n": fit["n"],
                "design_rank": fit["rank"],
                "residual_df": fit["residual_df"],
                "formula": (
                    "post_pct ~ pre_pct + pre_pct^2 + is_ai + is_human"
                    + (" + section_verbal" if scope == "combined" else "")
                ),
                "reference_group": "control",
                "covariance": "HC3 heteroskedasticity-consistent",
                "group_omnibus_wald_chi2": wald,
                "group_omnibus_df": 2,
                "group_omnibus_p_two_sided": float(stats.chi2.sf(wald, 2)),
            }
        )

    return {
        "raw_groups": raw_groups,
        "adjusted_groups": adjusted_groups,
        "adjusted_contrasts": adjusted_contrasts,
        "models": models,
    }


def _welch_tost(
    ai: np.ndarray, human: np.ndarray, fraction: float, alpha: float
) -> dict[str, Any]:
    n_ai, n_human = len(ai), len(human)
    sd_ai, sd_human = ai.std(ddof=1), human.std(ddof=1)
    pooled_sd = math.sqrt(
        ((n_ai - 1) * sd_ai**2 + (n_human - 1) * sd_human**2) / (n_ai + n_human - 2)
    )
    gap = float(ai.mean() - human.mean())
    se = math.sqrt(sd_ai**2 / n_ai + sd_human**2 / n_human)
    dof = se**4 / (
        (sd_ai**2 / n_ai) ** 2 / (n_ai - 1)
        + (sd_human**2 / n_human) ** 2 / (n_human - 1)
    )
    margin = fraction * pooled_sd
    p_lower = float(stats.t.sf((gap + margin) / se, dof))
    p_upper = float(stats.t.cdf((gap - margin) / se, dof))
    critical = float(stats.t.ppf(1.0 - alpha, dof))
    return {
        "n_ai": n_ai,
        "n_human": n_human,
        "ai_minus_human_raw_gap_pp": gap,
        "pooled_gain_sd_pp": pooled_sd,
        "margin_fraction_pooled_sd": fraction,
        "equivalence_margin_pp": margin,
        "welch_se": se,
        "welch_df": float(dof),
        "tost_alpha": alpha,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": max(p_lower, p_upper),
        "equivalence_ci_level": 1.0 - 2.0 * alpha,
        "equivalence_ci_low_pp": gap - critical * se,
        "equivalence_ci_high_pp": gap + critical * se,
        "equivalent_at_registered_alpha": max(p_lower, p_upper) < alpha,
    }
