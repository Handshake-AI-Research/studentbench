"""Post-hoc equivalence across the paper's finite subgroup robustness grid.

Non-sensitive memberships are computed from released pretreatment records.
Geographic and calendar-cohort tests use unlinked counts, means and sample
variances supplied in 6_population_demographics. These are sufficient for the
identical Welch/TOST calculations without individual locations or dates.

Definitions use membership and pretreatment variables only. Eligible groups
require >=20 AI and >=20 human sessions, with >=5 of each in each section.
Overlapping groups are dependent; their pass fraction is descriptive.
"""

from pathlib import Path
import csv
import hashlib
import json
import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .data import Dataset, load_sessions, sha256
from .journal import Journal, write_json, write_csv

AGGREGATE_FILE = "6_population_demographics/geography_equivalence_aggregates.csv"
COLLECTION_AGGREGATE_FILE = (
    "6_population_demographics/collection_cohort_equivalence_aggregates.csv"
)
GRID_COUNTS_FILE = "6_population_demographics/equivalence_robustness_grid_counts.csv"
FIELDS = [
    "student_survey_education_level",
    "student_survey_education_major",
    "student_survey_has_taken_sat",
    "student_survey_prior_tutoring_hours_category",
    "assessment_form_order",
    "study_participation_attempt_number",
    "pretest_irt_theta",
    "pretest_time_on_items_seconds",
    "pretest_wall_clock_seconds",
    "pretest_p90_inter_submission_seconds",
    "pretest_inter_submission_interval_count",
    "experiment_run_id",
    "human_tutor_id",
]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def tost(a, h, margin):
    return tost_from_moments(
        len(a),
        float(np.mean(a)),
        float(np.var(a, ddof=1)),
        len(h),
        float(np.mean(h)),
        float(np.var(h, ddof=1)),
        margin,
    )


def tost_from_moments(na, ma, va, nh, mh, vh, margin):
    gap = ma - mh
    se = math.sqrt(va / na + vh / nh)
    df = se**4 / ((va / na) ** 2 / (na - 1) + (vh / nh) ** 2 / (nh - 1))
    lo = float(stats.t.sf((gap + margin) / se, df))
    hi = float(stats.t.cdf((gap - margin) / se, df))
    crit = float(stats.t.ppf(0.95, df))
    sd = math.sqrt(((na - 1) * va + (nh - 1) * vh) / (na + nh - 2))
    return dict(
        n_ai=na,
        n_human=nh,
        ai_mean_gain_pp=ma,
        human_mean_gain_pp=mh,
        ai_minus_human_pp=gap,
        welch_se=se,
        welch_df=df,
        pooled_sd_pp=sd,
        margin_pp=margin,
        p_lower=lo,
        p_upper=hi,
        p_tost=max(lo, hi),
        ci90_low_pp=gap - crit * se,
        ci90_high_pp=gap + crit * se,
        equivalent=max(lo, hi) < 0.05,
    )


def read_aggregate_moments(data_dir, relative_path):
    fields = [
        "ai_session_count",
        "ai_mean_gain_pp",
        "ai_sample_variance_gain_pp2",
        "human_session_count",
        "human_mean_gain_pp",
        "human_sample_variance_gain_pp2",
    ]
    with (data_dir / relative_path).open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise ValueError(f"Unexpected aggregate schema: {relative_path}")
        rows = []
        for raw in reader:
            row = {
                k: int(v) if k.endswith("_session_count") else float(v)
                for k, v in raw.items()
            }
            if not all(math.isfinite(v) for v in row.values()):
                raise ValueError(f"Non-finite aggregate: {relative_path}")
            for kind in ["ai", "human"]:
                if (
                    row[kind + "_session_count"] < 20
                    or row[kind + "_sample_variance_gain_pp2"] <= 0
                ):
                    raise ValueError(f"Ineligible aggregate: {relative_path}")
            rows.append(row)
    if not rows or len({canonical(r) for r in rows}) != len(rows):
        raise ValueError(f"Missing or duplicate aggregates: {relative_path}")
    return rows


def geography_inputs(data_dir):
    rows = read_aggregate_moments(data_dir, AGGREGATE_FILE)
    count_fields = [
        "analysis_axis",
        "planned_definitions",
        "unique_groups",
        "duplicate_definitions",
        "eligible_unique_groups",
        "ineligible_unique_groups",
    ]
    with (data_dir / GRID_COUNTS_FILE).open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != count_fields:
            raise ValueError("Unexpected original-grid count schema")
        counts = []
        for raw in reader:
            row = {k: v if k == "analysis_axis" else int(v) for k, v in raw.items()}
            if (
                any(row[k] < 0 for k in count_fields[1:])
                or row["planned_definitions"]
                != row["unique_groups"] + row["duplicate_definitions"]
                or row["unique_groups"]
                != row["eligible_unique_groups"] + row["ineligible_unique_groups"]
            ):
                raise ValueError("Inconsistent original-grid counts")
            counts.append(row)
    if not counts or len({r["analysis_axis"] for r in counts}) != len(counts):
        raise ValueError("Missing or duplicate original-grid axes")
    totals = {k: sum(r[k] for r in counts) for k in count_fields[1:]}
    return rows, counts, totals


def geography_tasks(data_dir):
    rows, _, _ = geography_inputs(data_dir)
    return [
        dict(
            task_id="geography_aggregate_%04d" % (i + 1),
            axis="geography_aggregates",
            label="Unlinked geographic aggregate %d" % (i + 1),
            definition={
                "aggregate_file": AGGREGATE_FILE,
                "row_ordinal": i + 1,
                "group_definition_withheld_for_privacy": True,
            },
            n_total=r["ai_session_count"] + r["human_session_count"],
            n_ai=r["ai_session_count"],
            n_human=r["human_session_count"],
            eligible=True,
            ineligibility_reasons=[],
            duplicate_of=None,
            aggregate_moments=r,
        )
        for i, r in enumerate(rows)
    ]


def collection_cohort_tasks(data_dir):
    """Recompute the eligible calendar-cutoff test without revealing its dates."""
    rows = read_aggregate_moments(data_dir, COLLECTION_AGGREGATE_FILE)
    return [
        dict(
            task_id=f"collection_cohort_aggregate_{i + 1:04d}",
            axis="collection_wave",
            label=f"Unlinked collection-cohort aggregate {i + 1}",
            definition={
                "aggregate_file": COLLECTION_AGGREGATE_FILE,
                "row_ordinal": i + 1,
                "group_definition_withheld_for_privacy": True,
            },
            n_total=r["ai_session_count"] + r["human_session_count"],
            n_ai=r["ai_session_count"],
            n_human=r["human_session_count"],
            eligible=True,
            ineligibility_reasons=[],
            duplicate_of=None,
            aggregate_moments=r,
        )
        for i, r in enumerate(rows)
    ]


def aggregate_tost(r, margin):
    return tost_from_moments(
        r["ai_session_count"],
        r["ai_mean_gain_pp"],
        r["ai_sample_variance_gain_pp2"],
        r["human_session_count"],
        r["human_mean_gain_pp"],
        r["human_sample_variance_gain_pp2"],
        margin,
    )


def build_grid(data, data_dir):
    n = len(data)
    sec = np.array([x["section"] for x in data])
    kind = np.array([x["kind"] for x in data])
    tasks = []
    seen = {}

    def add(axis, label, mask, definition):
        mask = np.asarray(mask, dtype=bool)
        tokens = sorted(data[i]["token"] for i in np.flatnonzero(mask))
        membership = digest("".join(tokens).encode())
        task_id = digest(canonical([axis, label, definition]).encode())[:24]
        counts = {
            f"{a}_{b}": int(np.sum(mask & (sec == a) & (kind == b)))
            for a in ["quant", "verbal"]
            for b in ["ai", "human"]
        }
        na = counts["quant_ai"] + counts["verbal_ai"]
        nh = counts["quant_human"] + counts["verbal_human"]
        reasons = []
        if na < 20:
            reasons.append("n_ai_below_20")
        if nh < 20:
            reasons.append("n_human_below_20")
        if any(v < 5 for v in counts.values()):
            reasons.append("fewer_than_5_per_treatment_in_either_section")
        duplicate = seen.get(membership)
        seen.setdefault(membership, task_id)
        tasks.append(
            dict(
                task_id=task_id,
                axis=axis,
                label=label,
                definition=definition,
                membership_sha256=membership,
                n_total=na + nh,
                n_ai=na,
                n_human=nh,
                section_group_counts=counts,
                eligible=not reasons,
                ineligibility_reasons=reasons,
                duplicate_of=duplicate,
                mask=mask,
            )
        )

    add(
        "baseline",
        "full final AI and human cohort",
        np.ones(n, dtype=bool),
        {"all_final_tutored_assignments": True},
    )
    categoricals = [
        "student_survey_education_level",
        "student_survey_education_major",
        "student_survey_has_taken_sat",
        "student_survey_prior_tutoring_hours_category",
        "assessment_form_order",
    ]
    for field in categoricals:
        values = np.array(
            [str(x[field]) if x[field] is not None else "<missing>" for x in data]
        )
        for value in sorted(set(values)):
            add(
                "metadata_category",
                field + " = " + value,
                values == value,
                {"field": field, "equals": value, "missing_is_own_category": True},
            )
        add(
            "metadata_complete_case",
            field + " observed",
            values != "<missing>",
            {"field": field, "not_missing": True},
        )
    education = np.array(
        [
            str(x["student_survey_education_level"])
            if x["student_survey_education_level"] is not None
            else "<missing>"
            for x in data
        ]
    )
    education_groups = {
        "secondary_school": lambda v: v in ["9", "10", "11", "12", "middle_school"],
        "undergraduate": lambda v: v.startswith("undergrad_"),
        "graduate": lambda v: v.startswith("grad_"),
        "not_currently_in_school": lambda v: v == "post12_not_in_school",
        "undergrad_years_1_or_2": lambda v: v in ["undergrad_1", "undergrad_2"],
        "undergrad_years_3_or_higher": lambda v: (
            v in ["undergrad_3", "undergrad_4", "undergrad_5plus"]
        ),
        "undergraduate_or_graduate": lambda v: v.startswith(("undergrad_", "grad_")),
    }
    for label, pred in education_groups.items():
        add(
            "education_group",
            label,
            [pred(v) for v in education],
            {
                "education_group": label,
                "raw_categories": sorted(set(v for v in education if pred(v))),
            },
        )
    prior = np.array(
        [str(x["student_survey_prior_tutoring_hours_category"]) for x in data]
    )
    add(
        "prior_tutoring",
        "any reported prior tutoring",
        np.isin(prior, ["lt10", "10to50", "gt50"]),
        {
            "field": "student_survey_prior_tutoring_hours_category",
            "in": ["lt10", "10to50", "gt50"],
        },
    )
    score = np.array([x["pre_score"] for x in data])
    for cutoff in range(5, 25):
        for op in ["at_least", "at_most"]:
            add(
                "raw_pretest_score",
                f"{op} {cutoff} of 27",
                score >= cutoff if op == "at_least" else score <= cutoff,
                {"pretest_score_points": {op: cutoff}},
            )
    # All thresholds depend on pretreatment values only, computed in pooled tutored
    # assignments within each instrument (and form order for that specified axis).
    for field, byform in [
        ("pre_score", False),
        ("pre_score", True),
        ("pretest_irt_theta", False),
    ]:
        axis = (
            "percentile_"
            + field
            + ("_within_section_form" if byform else "_within_section")
        )
        values = np.array([float(x[field]) for x in data])
        strata = [
            x["section"] + (":" + x["assessment_form_order"] if byform else "")
            for x in data
        ]
        strata = np.array(strata)
        thresholds = {
            group: {
                str(q): float(np.quantile(values[strata == group], q, method="linear"))
                for q in [0.0] + [i / 20 for i in range(1, 20)] + [1.0]
            }
            for group in sorted(set(strata))
        }

        def boundary(q):
            return np.array([thresholds[g][str(q)] for g in strata])

        for q in [i / 20 for i in range(1, 20)]:
            for op in ["upper_tail", "lower_tail"]:
                add(
                    axis,
                    f"{op} at quantile {q:g}",
                    values >= boundary(q)
                    if op == "upper_tail"
                    else values <= boundary(q),
                    {
                        "quantile": q,
                        "tail": op,
                        "quantile_method": "linear",
                        "inclusive_cutoff": True,
                        "stratum_thresholds": {
                            g: t[str(q)] for g, t in thresholds.items()
                        },
                    },
                )
        for q in [i / 20 for i in range(1, 10)]:
            add(
                axis,
                f"central interval {q:g} to {1 - q:g}",
                (values >= boundary(q)) & (values <= boundary(round(1 - q, 2))),
                {
                    "quantile_interval": [q, round(1 - q, 2)],
                    "inclusive_cutoffs": True,
                    "stratum_thresholds": {
                        g: [t[str(q)], t[str(round(1 - q, 2))]]
                        for g, t in thresholds.items()
                    },
                },
            )
        for bins in [4, 5, 10]:
            for i in range(bins):
                lo = i / bins
                hi = (i + 1) / bins
                low = np.array(
                    [
                        np.quantile(values[strata == g], lo, method="linear")
                        for g in strata
                    ]
                )
                high = np.array(
                    [
                        np.quantile(values[strata == g], hi, method="linear")
                        for g in strata
                    ]
                )
                mask = (values >= low) & (
                    (values <= high) if i == bins - 1 else (values < high)
                )
                add(
                    axis,
                    f"{bins}-bin partition cell {i + 1}",
                    mask,
                    {
                        "quantile_interval": [lo, hi],
                        "lower_inclusive": True,
                        "upper_inclusive": i == bins - 1,
                        "ties_kept_together": True,
                    },
                )

    # The non-geographic two-way grid is rebuilt from released pretreatment data.
    # Geographic members are supplied separately as unlinked sufficient statistics.
    two_axes = {
        f: np.array([str(x[f]) if x[f] is not None else "<missing>" for x in data])
        for f in [
            "student_survey_education_level",
            "student_survey_education_major",
            "student_survey_has_taken_sat",
            "student_survey_prior_tutoring_hours_category",
        ]
    }
    for scorefield in ["pre_score", "pretest_irt_theta"]:
        v = np.array([float(x[scorefield]) for x in data])
        bounds = {
            g: [
                float(np.quantile(v[sec == g], i / 10, method="linear"))
                for i in range(11)
            ]
            for g in ["quant", "verbal"]
        }
        for i in range(10):
            low = np.array([bounds[g][i] for g in sec])
            high = np.array([bounds[g][i + 1] for g in sec])
            band = (v >= low) & ((v <= high) if i == 9 else (v < high))
            for field, values in two_axes.items():
                for value in sorted(set(values)):
                    add(
                        "two_way_decile_metadata",
                        f"{scorefield} decile {i + 1}; {field} = {value}",
                        band & (values == value),
                        {
                            "within_section_baseline_field": scorefield,
                            "decile": i + 1,
                            "metadata_field": field,
                            "metadata_value": value,
                            "interval_lower_inclusive": True,
                            "interval_upper_inclusive": i == 9,
                            "ties_kept_together": True,
                        },
                    )
    two_fields = list(two_axes)
    for i, left in enumerate(two_fields):
        for right in two_fields[i + 1 :]:
            lv, rv = two_axes[left], two_axes[right]
            for a, b in sorted(set(zip(lv, rv))):
                add(
                    "two_way_metadata_metadata",
                    f"{left} = {a}; {right} = {b}",
                    (lv == a) & (rv == b),
                    {
                        "left_field": left,
                        "left_value": a,
                        "right_field": right,
                        "right_value": b,
                        "domain": "all observed joint category combinations",
                    },
                )
    # Every contiguous interval on the 0,10,...,100 percentile lattice.
    for scorefield, byform in [
        ("pre_score", False),
        ("pre_score", True),
        ("pretest_irt_theta", False),
    ]:
        v = np.array([float(x[scorefield]) for x in data])
        strata = np.array(
            [
                x["section"] + (":" + x["assessment_form_order"] if byform else "")
                for x in data
            ]
        )
        bounds = {
            g: [
                float(np.quantile(v[strata == g], i / 10, method="linear"))
                for i in range(11)
            ]
            for g in sorted(set(strata))
        }
        for lo in range(10):
            for hi in range(lo + 1, 11):
                lower = np.array([bounds[g][lo] for g in strata])
                upper = np.array([bounds[g][hi] for g in strata])
                mask = (v >= lower) & ((v <= upper) if hi == 10 else (v < upper))
                add(
                    "contiguous_baseline_percentile_interval",
                    f"{scorefield} within section"
                    + ("/form" if byform else "")
                    + f" [{lo * 10},{hi * 10}] percentiles",
                    mask,
                    {
                        "baseline_field": scorefield,
                        "within_section": True,
                        "within_form": byform,
                        "lower_percentile": lo * 10,
                        "upper_percentile": hi * 10,
                        "lower_inclusive": True,
                        "upper_inclusive": hi == 10,
                        "quantile_method": "linear",
                        "ties_kept_together": True,
                    },
                )

    # Stricter pretreatment effort screens. These do not reapply posttreatment
    # protocol gates, relax the final cohort, or bring excluded participants back.
    for field in ["pretest_time_on_items_seconds", "pretest_wall_clock_seconds"]:
        value = np.array([float(x[field]) for x in data])
        for minutes in [10, 15, 20, 25, 30, 35, 40]:
            add(
                "pretest_effort",
                f"{field} >= {minutes} minutes",
                value >= 60 * minutes,
                {"field": field, "minimum_inclusive_seconds": 60 * minutes},
            )
    intervals = np.array(
        [int(x["pretest_inter_submission_interval_count"]) for x in data]
    )
    p90 = np.array([float(x["pretest_p90_inter_submission_seconds"]) for x in data])
    for count in [13, 20, 25, 26]:
        add(
            "pretest_timing_support",
            f"at least {count} supported intervals",
            intervals >= count,
            {"minimum_intervals_inclusive": count},
        )
    for minimum in [10, 15, 20, 30, 45, 60]:
        add(
            "pretest_rapid_response_sensitivity",
            f"p90 >= {minimum} seconds and >=13 intervals",
            (p90 >= minimum) & (intervals >= 13),
            {
                "pretest_p90_minimum_inclusive_seconds": minimum,
                "minimum_intervals_inclusive": 13,
            },
        )
    for minutes, seconds in [(15, 15), (20, 20), (25, 30)]:
        add(
            "pretest_joint_effort",
            f"item time >= {minutes} min; p90 >= {seconds} sec; >=13 intervals",
            np.array(
                [
                    float(x["pretest_time_on_items_seconds"]) >= 60 * minutes
                    for x in data
                ]
            )
            & (p90 >= seconds)
            & (intervals >= 13),
            {
                "item_time_minimum_seconds": 60 * minutes,
                "p90_minimum_seconds": seconds,
                "minimum_intervals": 13,
            },
        )
    attempts = np.array([int(x["study_participation_attempt_number"]) for x in data])
    add(
        "attempt_ordinal",
        "source attempt ordinal = 1",
        attempts == 1,
        {
            "field": "study_participation_attempt_number",
            "equals": 1,
            "not_a_person_deduplication": True,
        },
    )
    # The design-defined wave uses experiment membership, not collection dates.
    add(
        "collection_wave",
        "initial human-comparator collection waves",
        np.array(
            [
                (x["section"] == "quant" and x["experiment_run_id"] == 1)
                or (x["section"] == "verbal" and x["experiment_run_id"] == 4)
                for x in data
            ]
        ),
        {"quant_experiment_run_id": 1, "verbal_experiment_run_id": 4},
    )
    # AI-composition sensitivities retain every human comparator.
    arms = np.array([x["arm_id"] for x in data])
    aiarms = sorted(set(arms[kind == "ai"]))
    armsections = {a: set(sec[(kind == "ai") & (arms == a)]) for a in aiarms}
    for arm in aiarms:
        add(
            "leave_one_AI_configuration_out",
            "omit " + arm,
            (kind == "human") | (arms != arm),
            {"omit_AI_configuration": arm, "human_comparators": "all retained"},
        )
    shared = [a for a in aiarms if len(armsections[a]) == 2]
    add(
        "shared_AI_configurations",
        "AI identities fielded in both sections",
        (kind == "human") | np.isin(arms, shared),
        {"AI_identities": shared, "human_comparators": "all retained"},
    )
    providers = {
        "Google": [a for a in aiarms if a.startswith(("gemini", "gemma"))],
        "OpenAI": [a for a in aiarms if a.startswith("gpt")],
        "Anthropic": [a for a in aiarms if a.startswith(("opus", "sonnet"))],
        "Moonshot": [a for a in aiarms if a.startswith("kimi")],
    }
    for provider, aa in providers.items():
        add(
            "leave_one_AI_provider_out",
            "omit " + provider,
            (kind == "human") | ~np.isin(arms, aa),
            {
                "omitted_provider": provider,
                "omitted_AI_identities": aa,
                "human_comparators": "all retained",
            },
        )
    tutors = np.array([x["human_tutor_id"] for x in data])
    human_ids = sorted(set(tutors[kind == "human"]))
    if None in human_ids:
        raise ValueError(
            "Missing human tutor identity must be resolved before tutor omission grid."
        )
    for tutor in human_ids:
        add(
            "leave_one_human_tutor_out",
            "omit " + tutor,
            (kind == "ai") | (tutors != tutor),
            {"omit_anonymous_human_tutor_id": tutor, "AI_comparators": "all retained"},
        )
    return tasks + collection_cohort_tasks(data_dir) + geography_tasks(data_dir)


def run(data_dir: Path, output_dir: Path):
    """Recompute the baseline, 329 unique robustness tests and joint Holm family."""
    dataset = Dataset(data_dir)
    data_dir, output_dir = dataset.root, Path(output_dir)
    sessions = load_sessions(data_dir, output_dir / "inputs")
    data = []
    for row in sessions.to_dict("records"):
        if row["kind"] not in ("ai", "human"):
            continue
        profile = dataset.read_json(row["relative_profile_path"])
        data.append(
            dict(
                token=digest(row["student_id"].encode()),
                section=row["section"],
                kind=row["kind"],
                arm_id=row["arm_id"],
                gain=row["gain_pp"],
                pre_score=row["pre_score_points"],
                **{f: profile.get(f) for f in FIELDS},
            )
        )
    # This common primary margin is derived from the full AI/human cohort once;
    # secondary tests instead use a quarter of each subset's pooled SD.
    ai = np.array([row["gain"] for row in data if row["kind"] == "ai"])
    human = np.array([row["gain"] for row in data if row["kind"] == "human"])
    margin = 0.25 * tost(ai, human, 0)["pooled_sd_pp"]
    baseline = tost(ai, human, margin)
    tasks = build_grid(data, data_dir)
    _, grid_counts, totals = geography_inputs(data_dir)
    eligible = [
        task for task in tasks if task["duplicate_of"] is None and task["eligible"]
    ]
    if len(eligible) != totals["eligible_unique_groups"]:
        raise ValueError(
            "Reconstructed grid does not match released eligibility counts"
        )
    fingerprint = digest(
        (
            sha256(Path(__file__))
            + sha256(data_dir / AGGREGATE_FILE)
            + sha256(data_dir / COLLECTION_AGGREGATE_FILE)
            + sha256(data_dir / GRID_COUNTS_FILE)
            + "".join(sessions.input_signature)
        ).encode()
    )
    journal = Journal(output_dir / "subgroup_tests.jsonl")
    gains = np.array([row["gain"] for row in data])
    kinds = np.array([row["kind"] for row in data])
    results = []
    for task in tasks:
        result = journal.get(task["task_id"], fingerprint)
        if result is None:
            result = {k: v for k, v in task.items() if k != "mask"}
            if task["duplicate_of"] is not None:
                result["status"] = "duplicate_membership_not_retested"
            elif not task["eligible"]:
                result["status"] = "ineligible_counts_not_tested"
            else:
                if "aggregate_moments" in task:
                    primary = aggregate_tost(task["aggregate_moments"], margin)
                    secondary = aggregate_tost(
                        task["aggregate_moments"], 0.25 * primary["pooled_sd_pp"]
                    )
                else:
                    a = gains[task["mask"] & (kinds == "ai")]
                    h = gains[task["mask"] & (kinds == "human")]
                    if min(np.var(a, ddof=1), np.var(h, ddof=1)) <= 0:
                        raise ValueError("Nonpositive within-treatment variance")
                    primary = tost(a, h, margin)
                    secondary = tost(a, h, 0.25 * primary["pooled_sd_pp"])
                conclusion = (
                    "equivalence_established"
                    if primary["equivalent"]
                    else "not_established_inconclusive"
                    if primary["ci90_low_pp"] < margin
                    and primary["ci90_high_pp"] > -margin
                    else "outside_equivalence_margin"
                )
                result.update(
                    status="complete",
                    primary=primary,
                    primary_conclusion=conclusion,
                    secondary_within_slice_SD=secondary,
                )
            journal.save(task["task_id"], fingerprint, result)
        results.append(result)
    tested = [r for r in results if r["status"] == "complete"]
    variants = [r for r in tested if r["axis"] != "baseline"]
    corrected = multipletests(
        [r["primary"]["p_tost"] for r in variants], method="holm"
    )[1]
    for row, pvalue in zip(variants, corrected):
        row["primary_holm_p_tost"] = float(pvalue)
        row["holm_equivalent"] = pvalue < 0.05
    flat = [
        dict(
            task_id=r["task_id"],
            axis=r["axis"],
            label=r["label"],
            **r["primary"],
            primary_conclusion=r["primary_conclusion"],
            primary_holm_p_tost=r.get("primary_holm_p_tost"),
            secondary_p_tost=r["secondary_within_slice_SD"]["p_tost"],
            secondary_equivalent=r["secondary_within_slice_SD"]["equivalent"],
        )
        for r in tested
    ]
    write_csv(output_dir / "subgroup_estimates.csv", pd.DataFrame(flat))
    broad_families = {}
    for row in results:
        if row["duplicate_of"] is not None:
            continue
        axis = row["axis"]
        family = (
            "baseline"
            if axis == "baseline"
            else "AI_composition_omissions"
            if axis.startswith(("leave_one_AI", "shared_AI"))
            else "human_tutor_omissions"
            if axis == "leave_one_human_tutor_out"
            else "score_variants"
            if axis.startswith(("percentile", "raw_pretest", "contiguous_baseline"))
            else "timing_and_date_filters"
            if axis.startswith(("pretest_", "collection_", "attempt_"))
            else None
        )
        if family is None:
            continue
        count = broad_families.setdefault(
            family,
            dict(
                unique_memberships=0,
                eligible=0,
                valid=0,
                primary_equivalent=0,
                primary_inconclusive=0,
                primary_interval_outside_bound=0,
            ),
        )
        count["unique_memberships"] += 1
        count["eligible"] += int(row["eligible"])
        if row["status"] == "complete":
            count["valid"] += 1
            count["primary_equivalent"] += int(row["primary"]["equivalent"])
            count["primary_inconclusive"] += int(
                row["primary_conclusion"] == "not_established_inconclusive"
            )
            count["primary_interval_outside_bound"] += int(
                row["primary_conclusion"] == "outside_equivalence_margin"
            )
    # The original month groups were all ineligible. Retain their grid counts
    # without reconstructing or publishing calendar labels or memberships.
    for row in grid_counts:
        if row["analysis_axis"] == "collection_month":
            if row["eligible_unique_groups"] != 0:
                raise ValueError("Eligible calendar groups require aggregate moments")
            broad_families["timing_and_date_filters"]["unique_memberships"] += row[
                "unique_groups"
            ]
    summary = dict(
        complete=len(tested) == len(eligible),
        baseline=baseline,
        original_grid_counts=grid_counts,
        broad_families=broad_families,
        planned_definitions=totals["planned_definitions"],
        unique_memberships=totals["unique_groups"],
        duplicates=totals["duplicate_definitions"],
        ineligible_unique_memberships=totals["ineligible_unique_groups"],
        eligible_unique_memberships=len(eligible),
        valid_test_results=len(tested),
        primary_robustness_denominator=len(variants),
        primary_robustness_equivalence_numerator=sum(
            r["primary"]["equivalent"] for r in variants
        ),
        primary_robustness_not_established=sum(
            not r["primary"]["equivalent"] for r in variants
        ),
        primary_robustness_inconclusive=sum(
            r["primary_conclusion"] == "not_established_inconclusive" for r in variants
        ),
        primary_robustness_interval_outside_bound=sum(
            r["primary_conclusion"] == "outside_equivalence_margin" for r in variants
        ),
        primary_robustness_holm_equivalence_numerator=int(sum(corrected < 0.05)),
        secondary_within_slice_SD_passes_excluding_baseline=sum(
            r["secondary_within_slice_SD"]["equivalent"] for r in variants
        ),
    )
    write_json(output_dir / "summary.json", summary)
    return summary
