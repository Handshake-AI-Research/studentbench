"""Descriptive cohort, review and interaction counts quoted in the paper.

Age and participant overlap are read from the released population aggregates.
Individual ages and private cross-section identity links are not reconstructed.
"""

from pathlib import Path
import pandas as pd
from . import costs, conversation
from .data import Dataset, load_sessions
from .journal import write_json


def run(data_dir, output_dir, resource_dir=None, conversation_dir=None):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(data_dir)
    sessions = load_sessions(data_dir, output_dir / "sessions")
    resource_dir = Path(resource_dir) if resource_dir else output_dir / "resources"
    resource_dir.mkdir(parents=True, exist_ok=True)
    resources, _, _, _ = costs._extract_students(data_dir, resource_dir, force=False)
    cost_records, _ = costs._read_costs(data_dir)
    conversation_dir = (
        Path(conversation_dir) if conversation_dir else output_dir / "conversation"
    )
    conversation_dir.mkdir(parents=True, exist_ok=True)
    flags = conversation._compute_problem_flag_students(
        data_dir, conversation_dir / "flag_sessions.jsonl"
    )
    _, transcript_coverage = conversation._compute_conversation_sessions(
        data_dir, conversation_dir / "conversation_sessions.jsonl"
    )
    cohorts = sessions.groupby(["section", "kind"]).size().rename("n").reset_index()
    tutors = {}
    for section, block in sessions[sessions.kind == "human"].groupby("section"):
        sizes = block.groupby("human_tutor_id").size()
        tutors[section] = dict(
            tutors=len(sizes),
            sessions=len(block),
            minimum_students=int(sizes.min()),
            maximum_students=int(sizes.max()),
        )
    age = dataset.frame("6_population_demographics/age_distribution_summary.csv")
    overlap = dataset.frame("6_population_demographics/section_participant_overlap.csv")
    grades, reviews = [], []
    for section in ["quant", "verbal"]:
        base = f"2_human_ai_vs_ai_lesson_plan_rubric_grading_data/gre_{section}"
        reviews.append(
            pd.DataFrame(dataset.read_csv(base + "/human_grading_sessions.csv")).assign(
                section=section
            )
        )
        grades.append(
            pd.DataFrame(dataset.read_csv(base + "/rubric_grades.csv")).assign(
                section=section
            )
        )
    reviews, grades = (
        pd.concat(reviews, ignore_index=True),
        pd.concat(grades, ignore_index=True),
    )
    # One early review lacks the session link; other reviews of the same
    # pre-test supply it. Do not count that missing link as another subject.
    linked = reviews[reviews.student_id.ne("")]
    assert linked.groupby("review_subject_id").student_id.nunique().max() == 1
    main = (
        reviews.assign(in_main=reviews.student_id.isin(sessions.student_id))
        .groupby("review_subject_id")
        .in_main.any()
    )
    rich = reviews[
        (reviews.section == "quant")
        & reviews.student_id.str.contains(":opus-5-high-maxprompt:", regex=False)
    ]
    by_section = {}
    for section, block in reviews.groupby("section"):
        sizes = block.groupby("comparison_id").human_reviewer_anon_id.nunique()
        by_section[section] = dict(
            reviews=len(block),
            pretests=block.review_subject_id.nunique(),
            reviewers=block.human_reviewer_anon_id.nunique(),
            comparisons=block.comparison_id.nunique(),
            min_reviewers_per_comparison=int(sizes.min()),
            max_reviewers_per_comparison=int(sizes.max()),
        )
    pretests_per_reviewer = reviews.groupby(
        "human_reviewer_anon_id"
    ).review_subject_id.nunique()
    variants = (
        grades.groupby(["section", "question_variant_id"])
        .size()
        .rename("n")
        .reset_index()
    )
    ai = resources[resources.kind == "ai"]
    review_summary = dict(
        reviews=len(reviews),
        pretests=len(main),
        main_cohort_pretests=int(main.sum()),
        other_pretests=int((~main).sum()),
        reviewers=reviews.human_reviewer_anon_id.nunique(),
        reviewers_with_multiple_pretests=int((pretests_per_reviewer > 1).sum()),
        comparisons=reviews.comparison_id.nunique(),
        richer_prompt_quant_pretests=rich.review_subject_id.nunique(),
        richer_prompt_quant_reviews=len(rich),
        rubric_ratings=len(grades),
        planning_ratings=int(
            grades.rubric_question_id.isin(
                [
                    "relevant_concepts",
                    "concept_grouping",
                    "concept_prioritization",
                    "time_allocation",
                    "test_taking_strategies",
                ]
            ).sum()
        ),
        practice_ratings=int(
            grades.rubric_question_id.isin(
                [
                    "practice_alignment",
                    "difficulty_appropriateness",
                    "answer_key_accuracy",
                ]
            ).sum()
        ),
        by_section=by_section,
        question_variants=variants.to_dict("records"),
    )
    output = dict(
        complete=True,
        cohorts=cohorts.to_dict("records"),
        sessions=len(sessions),
        human_tutors=tutors,
        age_aggregates=age.to_dict("records"),
        participant_overlap={
            r.metric: int(r.n) for r in overlap.itertuples(index=False)
        },
        expert_reviews=review_summary,
        transcripts=transcript_coverage,
        cost_coverage=dict(
            sessions=len(cost_records),
            complete_estimates=int(
                cost_records.best_available_cost_method.ne(
                    "partial_evidence_lower_bound"
                ).sum()
            ),
            partial_lower_bounds=int(
                cost_records.best_available_cost_method.eq(
                    "partial_evidence_lower_bound"
                ).sum()
            ),
        ),
        interactions=dict(
            ai_sessions=len(ai),
            helpfulness_ratings_observed=int(ai.tutor_helpfulness_rating.notna().sum()),
            nonblank_tutor_messages=int(ai.n_nonblank_tutor_messages.sum()),
            nonblank_student_messages=int(ai.n_nonblank_student_messages.sum()),
            total_nonblank_chat_messages=int(
                (ai.n_nonblank_tutor_messages + ai.n_nonblank_student_messages).sum()
            ),
            answered_practice_problems=int(flags.n_answered.sum()),
            by_section=flags.groupby("section")[["n_answered", "n_flagged"]]
            .sum()
            .reset_index()
            .to_dict("records"),
        ),
    )
    write_json(output_dir / "summary.json", output)
    return output
