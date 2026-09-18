"""Table 13: expert preference with repeated reviewers and student pre-tests.

Fit weighted Bradley–Terry models directly from the eight released rubric
criteria. Two-way covariance is pre-test + reviewer - their intersection, with
finite-cluster corrections. Ties are excluded; strong preferences weigh two.
"""

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import expit
from .engagement import csv_rows, digest, write_json
from .engagement_models import adjustments

CRITERIA = [
    "relevant_concepts",
    "concept_grouping",
    "concept_prioritization",
    "time_allocation",
    "practice_alignment",
    "difficulty_appropriateness",
    "answer_key_accuracy",
    "test_taking_strategies",
]


def summarize(names, theta, covariance, df=None):
    reference = stats.norm if df is None else stats.t(df)
    critical = 1.96 if df is None else reference.ppf(0.975)
    se = np.sqrt(np.diag(covariance))
    models = [
        dict(
            model=name,
            ability=float(theta[i]),
            se=float(se[i]),
            lo=float(theta[i] - critical * se[i]),
            hi=float(theta[i] + critical * se[i]),
        )
        for i, name in enumerate(names)
    ]
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            difference = theta[i] - theta[j]
            error = np.sqrt(covariance[i, i] + covariance[j, j] - 2 * covariance[i, j])
            pairs.append(
                dict(
                    model_a=names[i],
                    model_b=names[j],
                    difference_a_minus_b=float(difference),
                    se=float(error),
                    p=float(2 * reference.sf(abs(difference / error))),
                )
            )
    for pair, p in zip(pairs, adjustments([r["p"] for r in pairs])["holm"]):
        pair["p_holm"] = p
    leader = names[int(np.argmax(theta))]
    ahead = [
        r
        for r in pairs
        if leader in [r["model_a"], r["model_b"]] and r["p_holm"] < 0.05
    ]
    return dict(
        models=models,
        pairwise_contrasts=pairs,
        leader=leader,
        leader_ahead_count_holm=len(ahead),
        df=df,
    )


def fit_network(events):
    events = events.loc[events.preference_score.ne(0)].copy()
    names = sorted(set(events.plan_a_model) | set(events.plan_b_model))
    index = {m: i for i, m in enumerate(names)}
    a, b = (
        events.plan_a_model.map(index).to_numpy(),
        events.plan_b_model.map(index).to_numpy(),
    )
    preference = events.preference_score.to_numpy(float)
    weights = np.abs(preference)
    winner, loser = np.where(preference > 0, a, b), np.where(preference > 0, b, a)
    n, k = len(events), len(names) - 1
    X = np.zeros((n, k))
    take = winner < k
    X[np.flatnonzero(take), winner[take]] = 1
    take = loser < k
    X[np.flatnonzero(take), loser[take]] = -1

    def likelihood(beta):
        difference = X @ beta
        return np.dot(weights, np.logaddexp(0, -difference)), X.T @ (
            -weights * expit(-difference)
        )

    fitted = optimize.minimize(
        likelihood,
        np.zeros(k),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-9, "maxiter": 2000},
    )
    beta = fitted.x
    assert np.max(np.abs(likelihood(beta)[1])) < 1e-4
    p = expit(X @ beta)
    scores = (weights * (1 - p))[:, None] * X
    bread = np.linalg.inv(X.T @ ((weights * p * (1 - p))[:, None] * X))
    center = np.vstack([np.eye(k), np.zeros((1, k))])
    center -= center.mean(axis=0)
    theta = center @ beta
    covariances = {}
    counts = {}
    for field in ["pretest", "reviewer", "intersection"]:
        groups, codes = np.unique(events[field].to_numpy(), return_inverse=True)
        totals = np.zeros((len(groups), k))
        np.add.at(totals, codes, scores)
        correction = len(groups) / (len(groups) - 1) * (n - 1) / (n - k)
        covariances[field] = (
            center @ (bread @ (totals.T @ totals) @ bread * correction) @ center.T
        )
        counts[field] = len(groups)
    two_way = (
        covariances["pretest"] + covariances["reviewer"] - covariances["intersection"]
    )
    eigenvalues, eigenvectors = np.linalg.eigh((two_way + two_way.T) / 2)
    # A centered model has one null eigenvalue. Record any substantive correction.
    projected = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    df = min(counts["pretest"], counts["reviewer"]) - 1
    return dict(
        n_decisive=n,
        cluster_counts=counts,
        negative_eigenvalues=int((eigenvalues < -1e-10).sum()),
        original=summarize(names, theta, covariances["pretest"]),
        sensitivity=summarize(names, theta, projected, df),
        sensitivity_normal=summarize(names, theta, projected),
    )


def run(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    pins = {}
    for section in ["quant", "verbal"]:
        folder = (
            data_dir
            / "2_human_ai_vs_ai_lesson_plan_rubric_grading_data"
            / ("gre_" + section)
        )
        for name in ["human_grading_sessions.csv", "rubric_grades.csv"]:
            pins[str((folder / name).relative_to(data_dir))] = digest(folder / name)
        sessions = pd.DataFrame(csv_rows(folder / "human_grading_sessions.csv"))
        grades = pd.DataFrame(csv_rows(folder / "rubric_grades.csv"))
        assert not sessions.grading_id.duplicated().any()
        assert not grades[["grading_id", "rubric_question_id"]].duplicated().any()
        keys = [
            "grading_id",
            "review_subject_id",
            "human_reviewer_anon_id",
            "lesson_plan_a_ai_model_preset_id",
            "lesson_plan_b_ai_model_preset_id",
        ]
        events = grades.merge(sessions[keys], on="grading_id", validate="many_to_one")
        assert len(events) == len(grades) and set(events.rubric_question_id) == set(
            CRITERIA
        )
        assert events.groupby("grading_id").size().eq(8).all()
        events["preference_score"] = pd.to_numeric(events.preference_score)
        assert events.preference_score.isin([-2, -1, 0, 1, 2]).all()
        events = events.rename(
            columns={
                "lesson_plan_a_ai_model_preset_id": "plan_a_model",
                "lesson_plan_b_ai_model_preset_id": "plan_b_model",
            }
        )
        events["pretest"] = section + ":" + events.review_subject_id
        events["reviewer"] = events.human_reviewer_anon_id
        events["intersection"] = events.pretest + "|" + events.reviewer
        frames.append(events)
    events = pd.concat(frames, ignore_index=True)
    assert len(events) == 16224
    identity = hashlib.sha256(
        json.dumps([pins, digest(__file__)], sort_keys=True).encode()
    ).hexdigest()
    results = {}
    for name, criteria in {
        "planning_combined": CRITERIA[:4] + [CRITERIA[7]],
        "practice_combined": CRITERIA[4:7],
        "common8_combined": CRITERIA,
    }.items():
        path = output_dir / (name + ".json")
        existing = json.loads(path.read_text()) if path.exists() else {}
        if existing.get("identity") == identity:
            result = existing["result"]
        else:
            result = fit_network(events.loc[events.rubric_question_id.isin(criteria)])
            write_json(path, dict(identity=identity, complete=True, result=result))
        results[name] = result
    summary = dict(
        status="complete",
        complete=True,
        input_sha256=pins,
        reviewers=int(events.reviewer.nunique()),
        reviews=int(events.grading_id.nunique()),
        reviewers_with_multiple_pretests=int(
            (events.groupby("reviewer").pretest.nunique() > 1).sum()
        ),
        results=results,
    )
    write_json(output_dir / "results.json", summary)
    return summary
