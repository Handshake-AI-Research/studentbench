"""OLS/HC3 models and the paper's 276-specification exploratory registry.

Each specification adjusts for starting score, exact AI model configuration,
and assessment form. Combined fits preserve section-specific forms and both
Sonnet versions. Tests are exploratory; Holm correction spans all 276 fits.
"""

from statistics import NormalDist
import hashlib
import math
import numpy as np
import pandas as pd
from scipy import linalg, stats

L = ["latency_mean_s", "latency_median_s"]
E = ["student_chat_messages", "student_messages_ge5_words", "student_words"]
P = [
    "practice_answered_distinct",
    "practice_submissions",
    "practice_final_credit_all",
    "practice_final_credit_closed",
    "practice_first_credit_closed",
]
A = ["student_messages_ge10_words", "student_nonspace_characters"]
TL = [
    "latency_mean_first30_request_complete_s",
    "latency_median_first30_request_complete_s",
]
SCOPES = ["quant", "verbal", "combined"]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def assert_close(actual, expected, label, probability=False):
    """Small p-values are checked relatively, never swallowed by 1e-8 atol."""
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), label
    tolerance = (
        dict(rtol=3e-7, atol=1e-300) if probability else dict(rtol=1e-8, atol=2e-9)
    )
    assert np.allclose(a, b, **tolerance), f"{label}: got {actual}; expected {expected}"


def registry():
    """Finite registry constructed independently of the stored result rows."""
    specs = {}

    def add(family, scope, x, y, edge):
        prefix = "temporal" if family == "temporal_raw" else family
        key = ":".join([prefix, scope, x, y])
        assert key not in specs
        specs[key] = (family, scope, x, y, edge)

    for scope in SCOPES:
        primary = (
            [("latency_engagement", x, y) for x in L for y in E]
            + [("engagement_practice", x, y) for x in E for y in P]
            + [("practice_gain", x, "gain_pp") for x in P]
        )
        for family in ["primary_raw", "secondary_log1p"]:
            for edge, x, y in primary:
                add(family, scope, x, y, edge)
        for x in L:
            for y in A:
                add("secondary_engagement", scope, x, y, "latency_engagement")
        for x in A:
            for y in P:
                add("secondary_engagement", scope, x, y, "engagement_practice")
        for x in TL:
            for y in E:
                add(
                    "temporal_raw",
                    scope,
                    x,
                    y + "_minutes30to60",
                    "early_latency_late_engagement",
                )
        for x in E:
            for y in P:
                add(
                    "temporal_raw",
                    scope,
                    x + "_first30",
                    y + "_minutes30to60",
                    "early_engagement_late_recorded_practice",
                )
        for x in P:
            add(
                "temporal_raw",
                scope,
                x + "_minutes30to60",
                "gain_pp",
                "late_recorded_practice_gain",
            )
    return specs


def vectors(sub, test):
    x = sub[test["x"]].to_numpy(float)
    y = sub[test["y"]].to_numpy(float)
    if test["family"] == "secondary_log1p":
        x = np.log1p(x)
        if test["y"] != "gain_pp":
            y = np.log1p(y)
    pre = sub.pre_pct.to_numpy(float)
    assert np.isfinite(np.column_stack([x, y, pre])).all()
    return x, y, pre


def scaled_fit(sub, test):
    x, y, pre = vectors(sub, test)
    scale = x.std(ddof=1)
    columns = [
        np.ones(len(sub)),
        (x - x.mean()) / scale,
        (pre - pre.mean()) / pre.std(ddof=1),
    ]
    forms = (
        sub.instrument + ":" + sub.form_order
        if test["scope"] == "combined"
        else sub.form_order
    )
    for values in [sub.arm_id, forms]:
        for level in sorted(values.unique())[1:]:
            columns.append((values.to_numpy() == level).astype(float))
    X = np.column_stack(columns)
    rank = np.linalg.matrix_rank(X)
    assert rank == X.shape[1]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    inv = np.linalg.inv(X.T @ X)
    residual = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, inv, X)
    assert (h < 1).all()
    cov = inv @ (X.T @ (((residual / (1 - h)) ** 2)[:, None] * X)) @ inv
    b, se = float(beta[1]), float(np.sqrt(cov[1, 1]))
    ci = np.array([b - stats.norm.ppf(0.975) * se, b + stats.norm.ppf(0.975) * se])
    units = (
        1.0
        if test["family"] == "secondary_log1p"
        else 100.0
        if test["x"].startswith("student_words")
        or test["x"] == "student_nonspace_characters"
        else 10.0
    )
    ysd = y.std(ddof=1)
    return {
        "n": len(sub),
        "n_models": int(sub.arm_id.nunique()),
        "rank": int(rank),
        "columns": X.shape[1],
        "estimate_per_x_sd": b,
        "se_per_x_sd": se,
        "ci95_per_x_sd": ci.tolist(),
        "x_sd": float(scale),
        "y_sd": float(ysd),
        "standardized_beta": b / ysd,
        "standardized_ci95": (ci / ysd).tolist(),
        "estimate_per_unit": b / scale,
        "ci95_per_unit": (ci / scale).tolist(),
        "report_units": units,
        "report_effect": b / scale * units,
        "report_ci95": (ci / scale * units).tolist(),
        "max_leverage": float(h.max()),
        "p_raw": float(2 * stats.norm.sf(abs(b / se))),
    }


def independent_qr_fit(sub, test):
    x, y, pre = vectors(sub, test)
    forms = (
        sub.instrument + ":" + sub.form_order
        if test["scope"] == "combined"
        else sub.form_order
    )
    dummies = pd.get_dummies(
        pd.DataFrame({"model": sub.arm_id, "form": forms}), drop_first=True
    ).to_numpy(float)
    X = np.column_stack([np.ones(len(sub)), pre, dummies, x])
    Q, R = linalg.qr(X, mode="economic")
    assert np.linalg.matrix_rank(R) == X.shape[1]
    coef = linalg.solve_triangular(R, Q.T @ y)
    residual = y - X @ coef
    leverage = (Q**2).sum(axis=1)
    influence = linalg.solve_triangular(R, Q.T)[-1]
    se = math.sqrt(float(np.sum((influence * residual / (1 - leverage)) ** 2)))
    b = float(coef[-1])
    z = NormalDist().inv_cdf(0.975)
    return {
        "estimate_per_unit": b,
        "se_per_unit": se,
        "ci95_per_unit": [b - z * se, b + z * se],
        "p_raw": math.erfc(abs(b / se) / math.sqrt(2)),
    }


def adjustments(values):
    """Direct order-statistic definitions, independent of stored correction code."""
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    m = len(values)
    holm, bh = [None] * m, [None] * m
    for rank, index in enumerate(order):
        holm[index] = min(1.0, max((m - j) * values[order[j]] for j in range(rank + 1)))
        bh[index] = min(
            1.0, min(m / (j + 1) * values[order[j]] for j in range(rank, m))
        )
    return {"holm": holm, "bh": bh}
