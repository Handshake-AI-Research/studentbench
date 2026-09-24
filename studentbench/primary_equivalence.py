"""Pooled AI–human equivalence and repeat-participant/tutor-omission checks.

Coefficients and fixed equivalence margins are estimated from released response
files. Within a section, public tutor aliases identify the dependence clusters.
For combined comparisons, global CR2 moment inputs encode dependence across
sections. Anonymous tutor aliases define tutor omissions; no direct identifiers
or participant-pair lists are included. See assets/analysis/README.md.
"""

from pathlib import Path
import hashlib
import json

import numpy as np
from scipy import stats

from .data import Dataset, load_sessions, sha256
from .journal import Journal, write_json


INPUT = Path(__file__).resolve().parents[1] / "assets/analysis/pooled_equivalence_cr2.json"
OMISSIONS = INPUT.with_name("leave_one_tutor_out_cr2.json")


def session_fingerprint(frame):
    """Bind aggregate moments to the public outcomes and design."""
    columns = ["student_id", "section", "kind", "arm_id", "human_tutor_id",
               "pre_pct", "post_pct", "form_order"]
    records = frame.sort_values("student_id")[columns].fillna({"human_tutor_id": ""}).to_dict("records")
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def design(frame, scope, weights):
    """Section-specific intercept, human effect, quadratic pretest, and form."""
    pre = (frame.pre_pct.to_numpy(float) - 50) / 25
    human = (frame.kind == "human").to_numpy(float)
    form = (frame.form_order == "QP").to_numpy(float)
    columns, contrast = [], []
    for section in (["quant", "verbal"] if scope == "combined" else [scope]):
        selected = (frame.section == section).to_numpy(float)
        columns.extend([selected, selected * human, selected * pre,
                        selected * pre**2, selected * form])
        contrast.extend([0, -weights[section] if scope == "combined" else -1, 0, 0, 0])
    return np.asarray(columns).T, np.asarray(contrast, float)


def cr2_moments(x, y, groups, contrast):
    """Sum CR2 working-model moments; never return per-cluster information.

    For cluster g, u_g = (I-H_gg)^(-1/2) X_g (X'X)^(-1)c and D_g=X_g'u_g.
    Summing global quadratic forms is sufficient for both the residual sandwich
    variance and the coefficient-specific Satterthwaite degrees of freedom.
    """
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Pooled-equivalence design is not full rank")
    bread = np.linalg.inv(x.T @ x)
    dimension = x.shape[1]
    outer = np.zeros((dimension, dimension))
    weighted_outer = np.zeros_like(outer)
    cross = np.zeros(dimension)
    response_square = norm_sum = norm_square_sum = 0.0
    for group in dict.fromkeys(groups):
        indices = np.flatnonzero(groups == group)
        block = x[indices]
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.eye(len(indices)) - block @ bread @ block.T
        )
        if eigenvalues.min() <= 1e-10:
            raise ValueError("CR2 adjustment has a singular cluster block")
        adjustment = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
        u = adjustment @ block @ bread @ contrast
        direction = block.T @ u
        response = float(u @ y[indices])
        norm = float(u @ u)
        product = np.outer(direction, direction)
        outer += product
        weighted_outer += norm * product
        cross += direction * response
        response_square += response**2
        norm_sum += norm
        norm_square_sum += norm**2
    return dict(
        design_outer_sum=outer.tolist(),
        norm_weighted_design_outer_sum=weighted_outer.tolist(),
        response_design_sum=cross.tolist(),
        response_square_sum=response_square,
        norm_sum=norm_sum,
        norm_square_sum=norm_square_sum,
        clusters=len(set(groups)),
    )


def fit(x, y, contrast, moments):
    """Re-estimate OLS and derive CR2 uncertainty from global moment sums."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Pooled-equivalence design is not full rank")
    bread = np.linalg.inv(x.T @ x)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    outer = np.asarray(moments["design_outer_sum"])
    cross = np.asarray(moments["response_design_sum"])
    weighted_outer = np.asarray(moments["norm_weighted_design_outer_sum"])
    variance = float(moments["response_square_sum"] - 2 * beta @ cross
                     + beta @ outer @ beta)
    trace = float(moments["norm_sum"] - np.trace(bread @ outer))
    squared_trace = float(moments["norm_square_sum"]
                         - 2 * np.trace(bread @ weighted_outer)
                         + np.trace(bread @ outer @ bread @ outer))
    if variance <= 0 or trace <= 0 or squared_trace <= 0:
        raise ValueError("Invalid CR2 variance or degrees-of-freedom moments")
    se, degrees = np.sqrt(variance), trace**2 / squared_trace
    estimate = float(contrast @ beta)
    return dict(
        estimate_pp=estimate, se_pp=float(se), df=float(degrees),
        ci90_pp=(estimate + np.array([-1, 1]) * stats.t.ppf(.95, degrees) * se).tolist(),
        ci95_pp=(estimate + np.array([-1, 1]) * stats.t.ppf(.975, degrees) * se).tolist(),
        clusters=moments["clusters"], n=len(y),
    )


def fixed_margins(sessions):
    result = {}
    for scope in ["quant", "verbal", "combined"]:
        frame = sessions if scope == "combined" else sessions[sessions.section == scope]
        ai = frame.loc[frame.kind == "ai", "gain_pp"].to_numpy(float)
        human = frame.loc[frame.kind == "human", "gain_pp"].to_numpy(float)
        result[scope] = float(.25 * np.sqrt(
            ((len(ai) - 1) * ai.var(ddof=1) + (len(human) - 1) * human.var(ddof=1))
            / (len(ai) + len(human) - 2)))
    return result


def run(data_dir, output_dir):
    output_dir = Path(output_dir)
    sessions = load_sessions(Path(data_dir), output_dir / "sessions")
    sessions = sessions[sessions.kind.isin(["ai", "human"])].copy()
    excluded = {row["student_id"] for row in Dataset(data_dir).read_csv(
        "6_population_demographics/repeat_participant_sessions.csv")}
    margins = fixed_margins(sessions)
    weights = {section: float((sessions.section == section).mean())
               for section in ["quant", "verbal"]}
    inputs = json.loads(INPUT.read_text())
    signature = sha256(Path(__file__)) + sha256(INPUT) + sha256(OMISSIONS) + session_fingerprint(sessions)
    signature += hashlib.sha256("\n".join(sorted(excluded)).encode()).hexdigest()
    journal = Journal(output_dir / "models.jsonl")
    results = {}
    for restricted in [False, True]:
        cohort = sessions[~sessions.student_id.isin(excluded)] if restricted else sessions
        for scope in ["combined", "quant", "verbal"]:
            key = ("exclude_repeat_" if restricted else "") + "adjusted_" + scope
            cached = journal.get(key, signature)
            if cached is not None:
                results[key] = cached
                continue
            frame = cohort if scope == "combined" else cohort[cohort.section == scope]
            x, contrast = design(frame, scope, weights)
            y = frame.post_pct.to_numpy(float)
            if scope == "combined":
                source = inputs["models"][key]
                if source["session_fingerprint"] != session_fingerprint(frame):
                    raise ValueError(f"{key}: public observations differ from CR2 moment inputs")
                moments = source["moments"]
                tutor_count = source["human_tutors"]
            else:
                groups = ["human:" + str(row.human_tutor_id) if row.kind == "human"
                          else "ai:" + row.student_id for row in frame.itertuples()]
                moments = cr2_moments(x, y, groups, contrast)
                tutor_count = frame.loc[frame.kind == "human", "human_tutor_id"].nunique()
            result = fit(x, y, contrast, moments)
            result.update(scope=scope, adjusted=True, clustering="connected",
                          n_ai=int((frame.kind == "ai").sum()),
                          n_human=int((frame.kind == "human").sum()),
                          n_human_tutors=int(tutor_count))
            result["margins"] = []
            for fraction in [.20, .25]:
                margin = margins[scope] * fraction / .25
                p = max(stats.t.sf((result["estimate_pp"] + margin) / result["se_pp"], result["df"]),
                        stats.t.cdf((result["estimate_pp"] - margin) / result["se_pp"], result["df"]))
                result["margins"].append(dict(margin_sd=fraction, margin_pp=margin, p_tost=float(p)))
            results[key] = journal.save(key, signature, result)
    # Anonymous public tutor aliases define the omissions. Global moments retain
    # dependence from repeated students without releasing their identity pairs.
    omissions = json.loads(OMISSIONS.read_text())["models"]
    for key, source in omissions.items():
        cached = journal.get(key, signature)
        if cached is not None:
            results[key] = cached
            continue
        aliases = source["omitted_public_tutor_aliases"]
        frame = sessions[~sessions.human_tutor_id.isin(aliases)]
        if source["session_fingerprint"] != session_fingerprint(frame):
            raise ValueError(f"{key}: public observations differ from CR2 moment inputs")
        x, contrast = design(frame, "combined", weights)
        value = fit(x, frame.post_pct.to_numpy(float), contrast, source["moments"])
        value.update(n_ai=int((frame.kind == "ai").sum()),
                     n_human=int((frame.kind == "human").sum()),
                     omitted_public_tutor_aliases=aliases)
        value["margins"] = []
        for fraction in [.20, .25]:
            margin = margins["combined"] * fraction / .25
            p = max(stats.t.sf((value["estimate_pp"] + margin) / value["se_pp"], value["df"]),
                    stats.t.cdf((value["estimate_pp"] - margin) / value["se_pp"], value["df"]))
            value["margins"].append(dict(margin_sd=fraction, margin_pp=margin, p_tost=float(p)))
        results[key] = journal.save(key, signature, value)
    result = dict(complete=True, status="complete", models=results,
                  section_weights=weights, fixed_margins_pp=margins,
                  dependence_coverage=inputs["coverage"])
    write_json(output_dir / "results.json", result)
    return result
