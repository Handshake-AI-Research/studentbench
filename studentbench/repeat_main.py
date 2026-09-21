"""Repeat-participation checks for learning, teaching and resource frontiers.

This complements Table 8's 276-fit engagement sensitivity. Membership comes
from the two public exclusion lists; no participant identity map is needed.
"""

from pathlib import Path
import hashlib
import pandas as pd
from scipy import stats

from . import costs, conversation, statistics, teaching
from .data import Dataset, load_sessions
from .journal import Journal, write_json, write_csv
from .sensitivity import welch_tost


def conversation_summary(frame):
    rows, tests = [], []
    for scope in ["quant", "verbal", "combined"]:
        block = frame if scope == "combined" else frame[frame.instrument == scope]
        for arm, group in block.groupby("arm"):
            values = group.strategy_score.to_numpy(float)
            half = stats.t.ppf(0.975, len(values) - 1) * stats.sem(values)
            rows.append(
                dict(
                    scope=scope,
                    arm=arm,
                    n=len(values),
                    mean=float(values.mean()),
                    lo=float(values.mean() - half),
                    hi=float(values.mean() + half),
                )
            )
        test = stats.f_oneway(
            *[g.strategy_score.to_numpy(float) for _, g in block.groupby("arm")],
            equal_var=False,
        )
        ai = block.loc[block.arm != "human", "strategy_score"].to_numpy(float)
        human = block.loc[block.arm == "human", "strategy_score"].to_numpy(float)
        tests.append(
            dict(
                scope=scope,
                n=len(block),
                welch_f=float(test.statistic),
                welch_p=float(test.pvalue),
                ai_mean=float(ai.mean()),
                human_mean=float(human.mean()),
                ai_human_p=float(stats.ttest_ind(ai, human, equal_var=False).pvalue),
            )
        )
    for key in ["welch_p", "ai_human_p"]:
        for row, p in zip(tests, teaching.holm([r[key] for r in tests])):
            row[key + "_holm3"] = p
    ranges = {}
    for provider, prefixes in [
        ("Anthropic", ("opus", "sonnet")),
        ("OpenAI", ("gpt",)),
        ("Google", ("gemma", "gemini")),
    ]:
        values = [
            r["mean"]
            for r in rows
            if r["scope"] == "combined"
            and r["arm"].startswith(prefixes)
            and r["arm"] != "gemini-3.6-flash-low"
        ]
        ranges[provider] = [min(values), max(values)]
    return dict(rows=rows, tests=tests, provider_ranges_for_figure2_models=ranges)


def run(data_dir, output_dir, resource_dir=None, conversation_dir=None):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(data_dir)
    excluded = {
        r["student_id"]
        for r in dataset.read_csv(
            "6_population_demographics/repeat_participant_sessions.csv"
        )
    }
    excluded_reviews = {
        r["review_subject_id"]
        for r in dataset.read_csv(
            "6_population_demographics/repeat_participant_review_subjects.csv"
        )
    }
    sessions = load_sessions(data_dir, output_dir / "sessions")
    resource_dir = Path(resource_dir) if resource_dir else output_dir / "resources"
    resource_dir.mkdir(parents=True, exist_ok=True)
    students, _, _, _ = costs._extract_students(data_dir, resource_dir, force=False)
    cost_inputs, _ = costs._read_costs(data_dir, resource_dir)
    conversation_dir = (
        Path(conversation_dir) if conversation_dir else output_dir / "conversation"
    )
    conversation_dir.mkdir(parents=True, exist_ok=True)
    dialogues, _ = conversation._compute_conversation_sessions(
        data_dir, conversation_dir / "conversation_sessions.jsonl"
    )
    dialogues["strategy_score"] = dialogues[list(conversation._DETECTORS)].mean(axis=1)
    # Transcript extraction uses section-local task directory names; translate
    # through the released profiles to the public study-session identifier.
    session_ids = {
        (r.section, Path(r.relative_profile_path).parents[1].name): r.student_id
        for r in sessions.itertuples(index=False)
    }
    dialogues["student_id"] = [
        session_ids[(r.instrument, r.student_id)]
        for r in dialogues.itertuples(index=False)
    ]
    events = teaching.load_ratings(data_dir)
    assert len(excluded) == 172 and len(excluded_reviews) == 26
    assert len(sessions) == len(students) == 2469 and len(dialogues) == 2274
    assert excluded <= set(sessions.student_id) and excluded_reviews <= set(
        events.review_subject_id
    )
    digest = hashlib.sha256()
    for frame in [sessions, students, cost_inputs, dialogues, events]:
        digest.update(frame.to_csv(index=False).encode())
    for module in [
        Path(__file__),
        Path(costs.__file__),
        Path(conversation.__file__),
        Path(teaching.__file__),
        Path(statistics.__file__),
    ]:
        digest.update(module.read_bytes())
    digest.update(
        repr((sorted(excluded), sorted(excluded_reviews), dataset.parameters)).encode()
    )
    signature = digest.hexdigest()
    journal = Journal(output_dir / "fits.jsonl")

    def task(key, function):
        value = journal.get(key, signature)
        if value is None:
            value = journal.save(key, signature, function())
        return value

    full_ai = sessions.loc[sessions.kind == "ai", "gain_pp"].to_numpy(float)
    full_human = sessions.loc[sessions.kind == "human", "gain_pp"].to_numpy(float)
    margin = statistics._welch_tost(full_ai, full_human, 0.25, 0.05)[
        "equivalence_margin_pp"
    ]
    analyses = {}
    for label in ["original", "exclude_repeat"]:
        restricted = label == "exclude_repeat"
        frame = (
            sessions[~sessions.student_id.isin(excluded)] if restricted else sessions
        )
        resource = (
            students[~students.student_id.isin(excluded)] if restricted else students
        )
        transcript = (
            dialogues[~dialogues.student_id.isin(excluded)] if restricted else dialogues
        )
        rating = (
            events[~events.review_subject_id.isin(excluded_reviews)]
            if restricted
            else events
        )
        destination = output_dir / label
        destination.mkdir(exist_ok=True)
        primary = task(
            label + ":primary",
            lambda: statistics._estimate_outcomes(frame.to_dict("records")),
        )

        def cost_fit():
            table, _ = costs._build_cost_rows(
                resource,
                cost_inputs[cost_inputs.student_id.isin(resource.student_id)],
                destination,
                dataset.parameters,
                force=False,
            )
            return table.to_dict("records")

        cost_rows = task(label + ":cost", cost_fit)
        write_csv(destination / "cost_rows.csv", pd.DataFrame(cost_rows))
        combined = [r for r in cost_rows if r["scope"] == "combined"]
        human_cost = next(
            r["cost_per_gain_pp"] for r in combined if r["arm_id"] == "human"
        )
        flash_cost = next(
            r["cost_per_gain_pp"]
            for r in combined
            if r["arm_id"] == "gemini-3.5-flash-low"
        )
        human = frame.loc[frame.kind == "human", "gain_pp"].to_numpy(float)
        individual = []
        for row in combined:
            arm = row["arm_id"]
            if arm == "human":
                continue
            fit = task(
                label + ":individual:" + arm,
                lambda arm=arm: welch_tost(
                    frame.loc[frame.arm_id == arm, "gain_pp"].to_numpy(float),
                    human,
                    margin,
                ),
            )
            individual.append(dict(arm_id=arm, **fit))
        latency = task(
            label + ":latency",
            lambda: costs._build_latency_rows(resource)[0].to_dict("records"),
        )
        experts = {}
        for group, criteria in [
            ("lesson_planning", teaching.CRITERIA[:4] + [teaching.CRITERIA[7]]),
            ("practice_problem_design", teaching.CRITERIA[4:7]),
        ]:
            experts[group] = task(
                label + ":" + group,
                lambda criteria=criteria: teaching.fit_bt(
                    rating[rating.rubric_question_id.isin(criteria)]
                ),
            )
        analyses[label] = dict(
            sessions=len(frame),
            by_section_kind=frame.groupby(["section", "kind"])
            .size()
            .rename("n")
            .reset_index()
            .to_dict("records"),
            adjusted_contrasts=primary["adjusted_contrasts"],
            individual_combined_equivalence=individual,
            cost=dict(
                human_per_pp=human_cost,
                gemini35flash_per_pp=flash_cost,
                human_to_flash_cost_ratio=human_cost / flash_cost,
                combined_frontier_models=sorted(
                    r["arm_id"] for r in combined if r["pareto_frontier"]
                ),
                ratio_bootstrap_valid=sum(
                    r["bootstrap_valid_draws"] for r in cost_rows
                ),
                ratio_bootstrap_planned=sum(r["bootstrap_draws"] for r in cost_rows),
            ),
            reply_time_frontier_models=sorted(
                r["arm_id"]
                for r in latency
                if r["scope"] == "combined" and r["pareto_frontier"]
            ),
            expert_preference=experts,
            conversation=task(
                label + ":conversation", lambda: conversation_summary(transcript)
            ),
        )
        write_json(destination / "results.json", analyses[label])
    before, after = analyses["original"], analyses["exclude_repeat"]
    ranks = {
        group: [r["model"] for r in before["expert_preference"][group]["models"]]
        == [r["model"] for r in after["expert_preference"][group]["models"]]
        for group in before["expert_preference"]
    }
    provider_ranges = after["conversation"]["provider_ranges_for_figure2_models"]
    checks = dict(
        expert_rankings_unchanged=ranks,
        cost_frontier_unchanged=before["cost"]["combined_frontier_models"]
        == after["cost"]["combined_frontier_models"],
        reply_time_frontier_unchanged=before["reply_time_frontier_models"]
        == after["reply_time_frontier_models"],
        provider_bands_remain_ordered=provider_ranges["Anthropic"][0]
        > provider_ranges["OpenAI"][1]
        and provider_ranges["OpenAI"][0] > provider_ranges["Google"][1],
    )
    result = dict(
        complete=True,
        excluded_sessions=len(excluded),
        excluded_review_subjects=len(excluded_reviews),
        analyses=analyses,
        checks=checks,
    )
    write_json(output_dir / "results.json", result)
    return result
