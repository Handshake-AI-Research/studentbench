"""Appendix prompt comparisons: exploratory pilot and later Quant cohorts.

The five-cell pilot is kept separate from the later nonconcurrent main/minimal
and expanded-prompt cohorts. The pilot adjusts for the actual plan generator
and clusters by released first-attempt identity; the later comparison uses HC3.
Neither analysis identifies a causal effect of prompt length.
"""

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices, dmatrix

from .data import Dataset, sha256
from .journal import Journal, plain, write_json, write_csv

SUP = "4_supplementary_and_excluded_research_data"
PILOT = (
    f"{SUP}/quant_exploratory_prompt_optimization_pilot/prompt_analysis_sessions.csv"
)
EXCLUDED = f"{SUP}/quant_exploratory_prompt_optimization_pilot/excluded_data/sessions_excluded_from_prompt_analysis.csv"


def run(data_dir: Path, output_dir: Path):
    """Recompute all prompt-comparison sessions, group estimates and contrasts."""
    data = Dataset(data_dir)
    ssot = data.root
    output_dir = Path(output_dir)
    journal = Journal(output_dir / "checkpoints.jsonl")
    paths = [ssot / PILOT, ssot / EXCLUDED, ssot / "collection_metadata.json"]
    for model in ("opus-5-high", "gemini-3.6-flash-low"):
        for folder in (
            ssot / f"1_main_leaderboard_data/gre_quant/ai_arm_{model}",
            ssot / f"{SUP}/quant_ai_max_prompt_runs/{model}-maxprompt",
        ):
            paths.extend(sorted(folder.glob("task_id_*/student_data/*")))
    signature = hashlib.sha256(
        (
            sha256(Path(__file__))
            + "".join(sha256(path) for path in paths if path.is_file())
        ).encode()
    ).hexdigest()
    done = {
        key: entry["value"]
        for key, entry in journal.rows.items()
        if entry["signature"] == signature
    }

    def emit(record_id, record_kind, **values):
        if record_id not in done:
            done[record_id] = journal.save(
                record_id,
                signature,
                dict(record_id=record_id, record_kind=record_kind, **values),
            )
        return done[record_id]

    def pin(path):
        return emit(
            "source:" + str(path.relative_to(ssot)),
            "source",
            relative_path=str(path.relative_to(ssot)),
            sha256=sha256(path),
        )

    pilot_pin = pin(ssot / PILOT)
    pin(ssot / EXCLUDED)
    pilot = []
    for raw in data.read_csv(PILOT):
        rid = "pilot:" + raw["student_id"]
        if rid in done:
            pilot.append(done[rid])
            continue
        pre = float(raw["pretest_score_points"])
        post = float(raw["posttest_score_points"])
        assert raw["prompt_analysis_included"] == "True" and all(
            raw[k] == "0" for k in ["is_admin", "is_test", "is_demo"]
        )
        assert (
            int(raw["pretest_maximum_score_points"])
            == int(raw["posttest_maximum_score_points"])
            == 27
        )
        assert abs(post - pre - float(raw["score_lift_points"])) < 1e-8
        row = emit(
            rid,
            "session",
            dataset="pilot",
            student_id=raw["student_id"],
            cluster=raw["first_attempt_student_id"],
            source_sha256=pilot_pin["sha256"],
            pre_points=pre,
            post_points=post,
            gain_pp=(post - pre) * 100 / 27,
            pre_pp=pre * 100 / 27,
            condition=raw["prompt_condition_id"],
            model=raw["ai_model_endpoint_id"],
            plan_model=raw["lesson_plan_generator_ai_model_endpoint_id"],
            form=raw["assessment_form_order"],
            plan_prompt=raw["lesson_plan_prompt_version_id"],
            tutor_prompt=raw["tutor_prompt_version_id"],
            fallback=raw["lesson_plan_generation_fallback_used"],
            bank=raw["assessment_bank_version_id"],
            attempt=int(raw["study_participation_attempt_number"]),
        )
        pilot.append(row)
    assert len(pilot) == 326 and len({r["student_id"] for r in pilot}) == 326
    excluded = data.read_csv(EXCLUDED)
    emit(
        "pilot_coverage",
        "coverage",
        dataset="pilot",
        expected=326,
        valid=len(pilot),
        invalid=0,
        source_rows_including_excluded=len(pilot) + len(excluded),
        source_exclusions=dict(
            Counter(r["prompt_analysis_exclusion_reason"] for r in excluded)
        ),
        unique_participant_clusters=len({r["cluster"] for r in pilot}),
        plan_tutor_identity_mismatches=sum(
            r["model"] != r["plan_model"] for r in pilot
        ),
        complete=True,
    )
    s5 = []
    for model, expanded_name in [
        ("opus-5-high", "opus-5-high-maxprompt"),
        ("gemini-3.6-flash-low", "gemini-3.6-flash-low-maxprompt"),
    ]:
        for condition, folder in [
            ("minimal", ssot / f"1_main_leaderboard_data/gre_quant/ai_arm_{model}"),
            ("expanded", ssot / f"{SUP}/quant_ai_max_prompt_runs/{expanded_name}"),
        ]:
            for path in sorted(folder.glob("task_id_*/student_data/student.json")):
                info = pin(path)
                rid = "s5:" + str(path.relative_to(ssot))
                # Recheck all referenced raw file hashes on resume too.
                score = {}
                for phase in ["pretest", "posttest"]:
                    response = path.parent / f"{phase}_responses.csv"
                    pin(response)
                    rr = list(csv.DictReader(response.open()))
                    assert len(rr) == 27
                    score[phase] = sum(int(float(r["is_correct"])) for r in rr)
                if rid in done:
                    s5.append(done[rid])
                    continue
                raw = json.loads(path.read_text())
                pre = float(raw["pretest_score_points"])
                post = float(raw["posttest_score_points"])
                assert pre == score["pretest"] and post == score["posttest"]
                assert abs(post - pre - float(raw["score_lift_points"])) < 1e-8
                row = emit(
                    rid,
                    "session",
                    dataset="s5",
                    student_id=raw["student_id"],
                    source_sha256=info["sha256"],
                    pre_points=pre,
                    post_points=post,
                    gain_pp=(post - pre) * 100 / 27,
                    pre_pp=pre * 100 / 27,
                    condition=condition,
                    model=model,
                    form=raw["assessment_form_order"],
                    plan_prompt=raw["lesson_plan_prompt_version_id"],
                    tutor_prompt=raw["tutor_prompt_version_id"],
                )
                s5.append(row)
    assert len(s5) == 332 and len({r["student_id"] for r in s5}) == 332
    emit(
        "s5_coverage",
        "coverage",
        dataset="s5",
        expected=332,
        valid=len(s5),
        invalid=0,
        assessment_files_recomputed=664,
        complete=True,
    )
    frames = {"pilot": pd.DataFrame(pilot), "s5": pd.DataFrame(s5)}
    frames["pilot"]["plan_style"] = frames["pilot"].condition.str.split("/").str[0]
    for dataset, frame in frames.items():
        groupcols = (
            [["condition"], ["model", "condition"], ["model", "plan_style"]]
            if dataset == "pilot"
            else [["model", "condition"]]
        )
        for cols in groupcols:
            for key, g in frame.groupby(cols, sort=True):
                key = key if isinstance(key, tuple) else (key,)
                detail = dict(zip(cols, key))
                rid = (
                    f"group:{dataset}:"
                    + ("plan_marginal:" if "plan_style" in cols else "")
                    + ":".join(key)
                )
                vals = g.gain_pp.to_numpy()
                se = stats.sem(vals)
                half = stats.t.ppf(0.975, len(vals) - 1) * se
                emit(
                    rid,
                    "group",
                    dataset=dataset,
                    **detail,
                    n=len(g),
                    mean_gain_pp=vals.mean(),
                    ci95_low=vals.mean() - half,
                    ci95_high=vals.mean() + half,
                    mean_pre_pp=g.pre_pp.mean(),
                    forms=g.form.value_counts().to_dict(),
                    models=g.model.value_counts().to_dict(),
                )

    def analyze(dataset, covariance):
        prefix = f"fit:{dataset}:{covariance}"
        if prefix + ":complete" in done:
            return
        frame = frames[dataset]
        formula = "gain_pp ~ C(condition)*C(model) + pre_pp + C(form)" + (
            " + C(plan_model)" if dataset == "pilot" else ""
        )
        fit = smf.ols(formula, frame).fit()
        if covariance == "cluster":
            fit = fit.get_robustcov_results(
                cov_type="cluster",
                groups=frame.cluster,
                use_correction=True,
                df_correction=True,
                use_t=False,
            )
        else:
            fit = fit.get_robustcov_results(cov_type="HC3", use_t=False)
        assert np.linalg.matrix_rank(fit.model.exog) == fit.model.exog.shape[1]
        names = fit.model.exog_names
        design_info = dmatrix(
            formula.split("~", 1)[1], frame, return_type="dataframe"
        ).design_info
        assert design_info.column_names == names, "Patsy/formula design mismatch"
        idx = [
            i
            for i, n in enumerate(names)
            if ":" in n and "C(condition)" in n and "C(model)" in n
        ]
        R = np.eye(len(names))[idx]
        wald = fit.wald_test(R, use_f=False, scalar=True)
        emit(
            prefix,
            "fit",
            dataset=dataset,
            covariance=covariance,
            formula=formula,
            n=int(fit.nobs),
            design_rank=len(names),
            n_clusters=frame.cluster.nunique() if covariance == "cluster" else None,
            parameters=dict(zip(names, fit.params)),
            interaction_df=len(idx),
            interaction_chi2=float(wald.statistic),
            interaction_p=float(wald.pvalue),
            covariance_matrix=fit.cov_params().tolist(),
            complete=True,
        )
        models = sorted(frame.model.unique())
        contrasts = {}

        def vec(model, condition):
            obj = {
                "model": model,
                "condition": condition,
                "pre_pp": float(frame.pre_pp.mean()),
                "form": sorted(frame.form.unique())[0],
            }
            if dataset == "pilot":
                obj["plan_model"] = sorted(frame.plan_model.unique())[0]
            return np.asarray(
                build_design_matrices([design_info], pd.DataFrame([obj]))[0]
            )[0]

        for model in models:
            contrasts[model] = vec(
                model, "max/max" if dataset == "pilot" else "expanded"
            ) - vec(model, "lean/lean" if dataset == "pilot" else "minimal")
        requests = [("within_model", m, v) for m, v in contrasts.items()]
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                requests.append(
                    (
                        "difference_between_models",
                        a + " minus " + b,
                        contrasts[a] - contrasts[b],
                    )
                )
        outputs = []
        for kind, label, vector in requests:
            test = fit.t_test(vector)
            ci = np.asarray(test.conf_int())[0]
            outputs.append(
                dict(
                    contrast_kind=kind,
                    label=label,
                    estimate_pp=float(np.asarray(test.effect).item()),
                    se=float(np.asarray(test.sd).item()),
                    ci95_low=float(ci[0]),
                    ci95_high=float(ci[1]),
                    p_nominal=float(np.asarray(test.pvalue).item()),
                )
            )
        ps = multipletests([r["p_nominal"] for r in outputs], method="holm")[1]
        for row, p in zip(outputs, ps):
            emit(
                prefix + ":contrast:" + row["label"],
                "contrast",
                dataset=dataset,
                covariance=covariance,
                **row,
                p_holm=float(p),
                holm_family_size=len(outputs),
            )
        emit(
            prefix + ":complete",
            "analysis_complete",
            dataset=dataset,
            covariance=covariance,
            contrasts=len(outputs),
            complete=True,
        )

    analyze("pilot", "cluster")
    analyze("pilot", "HC3")
    analyze("s5", "HC3")
    records = list(done.values())
    # The line-oriented file also makes every group and contrast easy to inspect.
    target = output_dir / "prompt_comparison_results.jsonl"
    with target.open("w") as handle:
        for row in records:
            handle.write(json.dumps(plain(row), allow_nan=False) + "\n")
    groups = [row for row in records if row["record_kind"] == "group"]
    write_csv(output_dir / "group_estimates.csv", pd.DataFrame(groups))
    summary = dict(
        complete=True,
        pilot_sessions=len(pilot),
        later_cohort_sessions=len(s5),
        fits=[row for row in records if row["record_kind"] == "fit"],
        contrasts=[row for row in records if row["record_kind"] == "contrast"],
        coverage=[row for row in records if row["record_kind"] == "coverage"],
    )
    write_json(output_dir / "summary.json", summary)
    return summary
