"""Learning outcomes from assessment responses, including every proficiency fit.

The same raw session table supplies the main result, section/domain summaries,
and robustness analyses. The released IRT theta values are measurements; the
original item calibration is not re-estimated here.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

from .data import load_sessions, sha256
from .journal import Journal, write_csv, write_json
from . import statistics, proficiency, domains as domain_estimates, irt


def load_topics(data_dir: Path, output_dir: Path, frame: pd.DataFrame) -> pd.DataFrame:
    journal = Journal(output_dir / "topics.jsonl")
    records = []
    for session in frame.to_dict("records"):
        sid = session["student_id"]
        signature = session["input_signature"] + sha256(Path(__file__))
        rows = journal.get(sid, signature)
        if rows is None:
            parent = (data_dir / session["relative_profile_path"]).parent
            pre = proficiency._assessment_topics(
                parent / "pretest_responses.csv", session["section"]
            )
            post = proficiency._assessment_topics(
                parent / "posttest_responses.csv", session["section"]
            )
            expected = {
                topic: n
                for section, topic, n in proficiency.DOMAIN_ORDER
                if section == session["section"]
            }
            if set(pre) != set(post) or set(pre) != set(expected):
                raise ValueError(f"{sid}: topic set differs")
            rows = []
            for topic, n in expected.items():
                if pre[topic][1] != n or post[topic][1] != n:
                    raise ValueError(f"{sid}: domain question count differs")
                rows.append(
                    {
                        k: session[k]
                        for k in [
                            "student_id",
                            "section",
                            "kind",
                            "arm_id",
                            "pre_score_points",
                        ]
                    }
                    | {
                        "topic": topic,
                        "n_items": n,
                        "pre_topic_pct": 100 * pre[topic][0] / n,
                        "post_topic_pct": 100 * post[topic][0] / n,
                        "gain_topic_pp": 100 * (post[topic][0] - pre[topic][0]) / n,
                    }
                )
            journal.save(sid, signature, rows)
        records.extend(rows)
    result = (
        pd.DataFrame(records)
        .sort_values(["section", "student_id", "topic"])
        .reset_index(drop=True)
    )
    write_csv(output_dir / "topic_rows.csv", result)
    return result


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
    return {
        "n": len(values),
        "mean_gain_pp": mean,
        "ci_low_pp": mean - half,
        "ci_high_pp": mean + half,
    }


def run(data_dir: Path, output_dir: Path):
    """Write fresh learning estimates and all associated table/figure inputs."""
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_sessions(data_dir, output_dir)
    topics = load_topics(data_dir, output_dir, frame)
    shared, coverage = proficiency._coverage(frame)
    primary = statistics._estimate_outcomes(frame.to_dict("records"))
    write_json(output_dir / "primary_results.json", primary)
    for key, value in primary.items():
        if isinstance(value, list):
            write_csv(output_dir / (key + ".csv"), pd.DataFrame(value))
    raw = proficiency._raw_outcomes(frame, shared)
    adjusted, adjusted_tests = proficiency._adjusted(frame, shared)
    write_csv(output_dir / "raw_arm_outcomes.csv", raw)
    write_csv(output_dir / "ancova_adjusted_arm_outcomes.csv", adjusted)
    write_json(output_dir / "adjusted_tests.json", adjusted_tests)

    categories, domains = [], []
    for section, topic, n in proficiency.DOMAIN_ORDER:
        block = topics[(topics.section == section) & (topics.topic == topic)]
        local = []
        for (arm, kind), group in block.groupby(["arm_id", "kind"]):
            row = {
                "section": section,
                "topic": topic,
                "n_items": n,
                "arm_id": arm,
                "kind": kind,
                "family": proficiency._family(arm),
                "arm_label": proficiency.MODEL_LABELS[arm],
                **mean_ci(group.gain_topic_pp),
            }
            local.append(row)
            categories.append(row)
        best = sorted(
            (r for r in local if r["kind"] == "ai"),
            key=lambda r: (-r["mean_gain_pp"], r["arm_id"]),
        )[0]
        domains.append(
            {
                "section": section,
                "topic": topic,
                "n_items": n,
                "best_ai": best,
                **{
                    kind: mean_ci(block.loc[block.kind == kind, "gain_topic_pp"])
                    for kind in ["ai", "human", "control"]
                },
            }
        )
    domain_table = pd.DataFrame(categories)
    domain_table["descriptive_rank"] = domain_table.groupby(["section", "topic"])[
        "mean_gain_pp"
    ].rank(ascending=False)
    write_csv(output_dir / "domain_arm_outcomes.csv", domain_table)
    topic_effects = topics.rename(
        columns={"section": "instrument", "gain_topic_pp": "gain_pp"}
    ).copy()
    topic_effects["arm_label"] = topic_effects.arm_id.map(proficiency.MODEL_LABELS)
    topic_effects["pre_correct"] = (
        topic_effects.pre_topic_pct * topic_effects.n_items / 100
    )
    topic_effects["post_correct"] = (
        topic_effects.post_topic_pct * topic_effects.n_items / 100
    )
    domain_effects, profile_tests = domain_estimates.build_effects(
        topic_effects.to_dict("records"), 0.95
    )
    write_csv(output_dir / "academic_field_effects.csv", domain_effects)
    write_json(output_dir / "domain_profile_tests.json", profile_tests)

    thresholds = proficiency._quartile_thresholds(frame)
    write_json(output_dir / "quartile_thresholds.json", thresholds)
    groups, contrasts = proficiency._proficiency_pooled_ai_control(frame, thresholds)
    write_csv(output_dir / "proficiency_quartile_group_outcomes.csv", groups)
    write_csv(output_dir / "proficiency_quartile_ai_control_contrasts.csv", contrasts)
    cells, global_tests, pairs, effects = (
        proficiency._configuration_topic_proficiency_frontier(topics, thresholds)
    )
    for name, table in [("cells", cells), ("pairwise", pairs), ("effects", effects)]:
        if not table.empty:
            write_csv(output_dir / f"topic_proficiency_configuration_{name}.csv", table)
    write_json(
        output_dir / "topic_proficiency_configuration_omnibus.json", global_tests
    )
    variants, irt_arms, irt_tests = irt.high_baseline_analysis(
        frame, output_dir / "irt"
    )
    for name, table in [
        ("irt_high_baseline_specification_family", variants),
        ("irt_top_quartile_quant_arm_outcomes", irt_arms),
    ]:
        write_csv(output_dir / (name + ".csv"), table)
    write_json(output_dir / "irt_high_baseline_statistics.json", irt_tests)

    figure = {
        "valid_sessions": len(frame),
        "topic_session_rows": len(topics),
        "domains": domains,
        "categories": categories,
        "combined_adjusted": adjusted[adjusted.scope == "combined"]
        .sort_values("display_order")
        .to_dict("records"),
        "pooled_combined": {
            kind: mean_ci(frame.loc[frame.kind == kind, "gain_pp"])
            for kind in ["ai", "human", "control"]
        },
        "ai_omnibus_tests": adjusted_tests,
        "model_coverage": coverage,
    }
    write_json(output_dir / "learning_figure_data.json", figure)
    summary = {
        "complete": True,
        "sessions": len(frame),
        "topic_rows": len(topics),
        "irt_specifications": len(variants),
        "shared_ai_models": len(shared),
        "source": "Released assessment responses and theta measurements",
    }
    write_json(output_dir / "summary.json", summary)
    return summary
