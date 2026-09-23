"""Weighted Bradley–Terry fits for the eight shared lesson-plan criteria."""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import expit

from . import conversation
from .data import Dataset
from .journal import write_json, write_csv, Journal


def fit_bt(events):
    """Analytic-gradient equivalent of pedagogy._fit_bt; same covariance/weights."""
    ev = events[events.preference_score != 0].copy()
    models = sorted(set(ev.plan_a_model) | set(ev.plan_b_model))
    k = len(models)
    idx = {m: i for i, m in enumerate(models)}
    a = ev.plan_a_model.map(idx).to_numpy()
    b = ev.plan_b_model.map(idx).to_numpy()
    pref = ev.preference_score.to_numpy(float)
    w = np.abs(pref)
    win = np.where(pref > 0, a, b)
    lose = np.where(pref > 0, b, a)
    X = np.zeros((len(ev), k - 1))
    rr = np.arange(len(ev))
    take = win < k - 1
    X[rr[take], win[take]] = 1
    take = lose < k - 1
    X[rr[take], lose[take]] = -1

    def fun(v):
        d = X @ v
        return np.dot(w, np.logaddexp(0, -d)), X.T @ (-w * expit(-d))

    opt = optimize.minimize(
        fun,
        np.zeros(k - 1),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-9, "maxiter": 2000},
    )
    v = opt.x
    pr = expit(X @ v)
    bread = np.linalg.pinv(X.T @ ((w * pr * (1 - pr))[:, None] * X))
    cluster, codes = np.unique(ev.cluster_id.astype(str), return_inverse=True)
    scores = np.zeros((len(cluster), k - 1))
    np.add.at(scores, codes, (w * (1 - pr))[:, None] * X)
    cov = bread @ (scores.T @ scores) @ bread
    cov *= len(cluster) / (len(cluster) - 1) * (len(ev) - 1) / (len(ev) - (k - 1))
    C = np.vstack([np.eye(k - 1), np.zeros((1, k - 1))])
    C -= C.mean(axis=0)
    theta = C @ v
    cov = C @ cov @ C.T
    se = np.sqrt(np.maximum(0, np.diag(cov)))
    modelrows = sorted(
        [
            dict(
                model=m,
                ability=float(theta[i]),
                se=float(se[i]),
                lo=float(theta[i] - 1.96 * se[i]),
                hi=float(theta[i] + 1.96 * se[i]),
            )
            for i, m in enumerate(models)
        ],
        key=lambda r: -r["ability"],
    )
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = theta[i] - theta[j]
            s = np.sqrt(cov[i, i] + cov[j, j] - 2 * cov[i, j])
            p = 2 * stats.norm.sf(abs(diff / s))
            pairs.append(
                {
                    "model_a": models[i],
                    "model_b": models[j],
                    "difference_a_minus_b": float(diff),
                    "se": float(s),
                    "z": float(diff / s),
                    "p": float(p),
                }
            )
    for r, p in zip(pairs, holm([r["p"] for r in pairs])):
        r["p_holm"] = p
    gradient_max = float(np.max(np.abs(fun(v)[1])))
    assert gradient_max < 1e-4, (opt.message, gradient_max)
    graph = np.zeros((k, k), bool)
    graph[win, lose] = True
    for q in range(k):
        graph |= graph[:, q, None] & graph[None, q, :]
    assert graph.all(), "Disconnected win graph"
    return {
        "models": modelrows,
        "pairwise_contrasts": pairs,
        "cluster_n": len(cluster),
        "n_decisive": len(ev),
        "n_all_grades": len(events),
        "n_ties": int((events.preference_score == 0).sum()),
        "weighted_decisive_n": float(w.sum()),
        "converged": True,
        "gradient_max": gradient_max,
        "win_graph_strongly_connected": True,
        "cluster_unit": "section-prefixed student pre-test",
        "method": "weighted BT; slight=1 strong=2; ties excluded; pre-test clustered sandwich; 1.96 normal intervals; Holm within each fitted network",
    }


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


def holm(values):
    order = np.argsort(values)
    corrected = np.empty(len(values))
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, values[index] * (len(values) - rank)))
        corrected[index] = previous
    return corrected.tolist()


def load_ratings(data_dir: Path):
    """Link raw rubric preferences to reviewed plans and underlying pre-tests."""
    data = Dataset(data_dir)
    frames = []
    for section in ["quant", "verbal"]:
        folder = (
            Path("2_human_ai_vs_ai_lesson_plan_rubric_grading_data") / f"gre_{section}"
        )
        sessions = data.frame(
            folder / "human_grading_sessions.csv", dtype=str, keep_default_na=False
        )
        grades = data.frame(
            folder / "rubric_grades.csv", dtype=str, keep_default_na=False
        )
        if (
            not sessions.grading_id.is_unique
            or grades.duplicated(["grading_id", "rubric_question_id"]).any()
        ):
            raise ValueError("Duplicate review or rubric grade")
        if set(grades.rubric_question_id) != set(CRITERIA):
            raise ValueError(
                "The release must contain exactly the eight shared rubric criteria"
            )
        if not grades.groupby("grading_id").size().eq(8).all():
            raise ValueError("A review is missing a shared rubric criterion")
        required = [
            "grading_id",
            "review_subject_id",
            "human_reviewer_anon_id",
            "lesson_plan_a_ai_model_preset_id",
            "lesson_plan_b_ai_model_preset_id",
        ]
        events = grades.merge(
            sessions[required], on="grading_id", validate="many_to_one", indicator=True
        )
        if set(events._merge) != {"both"}:
            raise ValueError("Unlinked rubric grades")
        events = events.rename(
            columns={
                "lesson_plan_a_ai_model_preset_id": "plan_a_model",
                "lesson_plan_b_ai_model_preset_id": "plan_b_model",
            }
        )
        events["preference_score"] = pd.to_numeric(
            events.preference_score, errors="raise"
        )
        if (
            not events.preference_score.isin([-2, -1, 0, 1, 2]).all()
            or (events.plan_a_model == events.plan_b_model).any()
        ):
            raise ValueError("Invalid pairwise rating")
        events["section"] = section
        events["cluster_id"] = section + ":" + events.review_subject_id
        frames.append(events.drop(columns="_merge"))
    return pd.concat(frames, ignore_index=True)


def run(data_dir: Path, output_dir: Path):
    """Fit all shared-criterion networks; derive conversation and dispute rates."""
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_ratings(data_dir)
    write_csv(output_dir / "rating_events.csv", events)
    signature = hashlib.sha256(
        events.to_csv(index=False).encode() + Path(__file__).read_bytes()
    ).hexdigest()
    journal = Journal(output_dir / "fits.jsonl")
    jobs = [
        (criterion, events[events.rubric_question_id == criterion])
        for criterion in CRITERIA
    ]
    for scope in ["quant", "verbal", "combined"]:
        block = events if scope == "combined" else events[events.section == scope]
        jobs.extend(
            [
                (f"common8_{scope}", block),
                (
                    f"practice_{scope}",
                    block[block.rubric_question_id.isin(CRITERIA[4:7])],
                ),
                (f"accuracy_{scope}", block[block.rubric_question_id == CRITERIA[6]]),
            ]
        )
    jobs += [
        (
            "planning_combined",
            events[events.rubric_question_id.isin(CRITERIA[:4] + [CRITERIA[7]])],
        ),
    ]
    fits = {}
    for name, block in jobs:
        result = journal.get(name, signature)
        if result is None:
            result = fit_bt(block)
            result["name"] = name
            journal.save(name, signature, result)
        write_json(output_dir / f"fit_{name}.json", result)
        fits[name] = result

    sessions, coverage = conversation._compute_conversation_sessions(
        data_dir, output_dir / "conversation_sessions.jsonl"
    )
    rules = list(conversation._DETECTORS)
    sessions["strategy_score"] = sessions[rules].mean(axis=1)
    write_csv(output_dir / "conversation_session_features.csv", sessions)
    prevalence = []
    for scope in ["quant", "verbal", "combined"]:
        block = (
            sessions if scope == "combined" else sessions[sessions.instrument == scope]
        )
        for arm, rows in block.groupby("arm"):
            mean = float(rows.strategy_score.mean())
            half = stats.t.ppf(0.975, len(rows) - 1) * stats.sem(rows.strategy_score)
            prevalence.append(
                {
                    "scope": scope,
                    "arm": arm,
                    "is_human": arm == "human",
                    "n": len(rows),
                    "mean_strategy_score": mean,
                    "mean_strategy_score_ci_low": mean - half,
                    "mean_strategy_score_ci_high": mean + half,
                    **{rule: float(rows[rule].mean()) for rule in rules},
                    **{rule + "_count": int(rows[rule].sum()) for rule in rules},
                }
            )
    prevalence = pd.DataFrame(prevalence)
    write_csv(output_dir / "conversation_strategy_prevalence.csv", prevalence)
    write_csv(
        output_dir / "conversation_by_section.csv",
        prevalence[prevalence.scope != "combined"],
    )
    write_csv(
        output_dir / "conversation_combined_leaderboard.csv",
        prevalence[prevalence.scope == "combined"],
    )
    write_json(output_dir / "conversation_coverage.json", coverage)

    flags = conversation._compute_problem_flag_students(
        data_dir, output_dir / "flag_sessions.jsonl"
    )
    settings = Dataset(data_dir).parameters["resampling"][
        "problem_flag_participant_bootstrap"
    ]
    flag_rows, omnibus = [], []
    for scope in ["quant", "verbal", "combined"]:
        block = flags if scope == "combined" else flags[flags.section == scope]
        rows, test = conversation._flag_scope_rows(
            block,
            scope,
            draws=int(settings["draws"]),
            seed=int(settings["seed"]) + (991 if scope == "combined" else 0),
            normalize_sonnet=scope == "combined",
        )
        flag_rows.append(rows)
        omnibus.append(test)
    for result, adjusted in zip(omnibus, holm([r["p_omnibus"] for r in omnibus])):
        result["p_holm_3"] = adjusted
    flag_rows = pd.concat(flag_rows, ignore_index=True)
    write_csv(output_dir / "flag_rates.csv", flag_rows)
    write_json(output_dir / "flag_tests.json", omnibus)
    # Correlate expert quality with student disputes within the original six-test family.
    correlations = []
    for scope in ["quant", "verbal", "combined"]:
        for construct in ["practice", "accuracy"]:
            rows = pd.DataFrame(fits[f"{construct}_{scope}"]["models"]).merge(
                flag_rows[flag_rows.scope == scope],
                left_on="model",
                right_on="model_preset",
                validate="one_to_one",
            )
            rho, p = stats.spearmanr(rows.ability, -rows.flags_per_100_answered)
            x, y = (
                stats.rankdata(rows.ability),
                stats.rankdata(-rows.flags_per_100_answered),
            )
            x -= x.mean()
            y -= y.mean()
            denominator = np.linalg.norm(x) * np.linalg.norm(y)
            rng = np.random.default_rng(20260912)
            extreme = 0
            for start in range(0, 99999, 5000):
                permutations = np.array(
                    [rng.permutation(y) for _ in range(min(5000, 99999 - start))]
                )
                extreme += int(
                    np.sum(np.abs(permutations @ x / denominator) >= abs(rho) - 1e-12)
                )
            correlations.append(
                {
                    "scope": scope,
                    "construct": construct,
                    "n_models": len(rows),
                    "k": len(rows),
                    "rho": float(rho),
                    "p": (extreme + 1) / 100000,
                    "spearman_rho": float(rho),
                    "p_asymptotic": float(p),
                    "p_permutation": (extreme + 1) / 100000,
                }
            )
            write_json(output_dir / "quality_dispute_correlations.json", correlations)
    for row, p in zip(correlations, holm([r["p_permutation"] for r in correlations])):
        row["p_holm_6"] = p
    write_json(output_dir / "quality_dispute_correlations.json", correlations)
    write_json(output_dir / "flag_expert_rank_agreement.json", correlations)
    summary = {
        "complete": True,
        "ratings": len(events),
        "networks": len(fits),
        "conversation_sessions": len(sessions),
        "flag_sessions": len(flags),
    }
    write_json(output_dir / "summary.json", summary)
    return summary
