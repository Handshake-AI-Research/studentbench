"""Check the paper's empirical prose against freshly computed analysis outputs.

The target file specifies printed precision, inequalities and named rankings.
It is used only for comparison: collect_metrics never reads expected results.
Protocol choices and cited external facts are documented separately rather than
presented as estimates reproduced from participant observations.
"""

from collections import Counter
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd

from .data import sha256
from .journal import Journal, plain, write_json


EXPECTED = Path(__file__).resolve().parents[1] / "verification/paper_results_expected.json"


def collect_metrics(analysis_dir, data_dir=None):
    root = Path(analysis_dir)
    metrics, sources = {}, {}

    def read(name):
        path = root / name
        sources[name] = sha256(path)
        value = json.loads(path.read_text())
        if isinstance(value, dict) and value.get("complete") is False:
            raise ValueError(f"Incomplete analysis: {name}")
        return value

    def table(name):
        path = root / name
        sources[name] = sha256(path)
        return pd.read_csv(path)

    def put(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                put(prefix + "." + str(key), child)
        else:
            metrics[prefix] = plain(value)

    study = read("study_summary/summary.json")
    put("study", study)
    for row in study["age_aggregates"]:
        put("age." + row["population"].lower(), row)
    for section in ["quant", "verbal"]:
        put("study.sessions_" + section,
            sum(row["n"] for row in study["cohorts"] if row["section"] == section))
    for row in study["interactions"]["by_section"]:
        put("practice." + row["section"], row)
    variants = study["expert_reviews"]["question_variants"]
    for suffix in ["format_weaknesses", "test_taking_strategies"]:
        put("rubric.quant_strategy_" + suffix,
            sum(row["n"] for row in variants if row["section"] == "quant"
                and row["question_variant_id"].endswith(suffix)))

    pooled = read("primary_equivalence/results.json")
    put("pooled", pooled["models"])
    for name, row in pooled["models"].items():
        for margin in row["margins"]:
            label = "primary" if margin["margin_sd"] == .25 else "narrow"
            put("pooled." + name + "." + label, margin)
        put("pooled." + name + ".equivalent", row["margins"][1]["p_tost"] < .05)
    put("pooled.section_weights", pooled["section_weights"])
    put("pooled.dependence_coverage", pooled["dependence_coverage"])
    omitted = [row for key, row in pooled["models"].items() if key.startswith("leave_one_tutor_out_")]
    put("pooled.leave_one_tutor_out.fits", len(omitted))
    put("pooled.leave_one_tutor_out.passing", sum(row["margins"][1]["p_tost"] < .05 for row in omitted))
    put("pooled.fixed_margins", pooled["fixed_margins_pp"])
    put("interaction", read("condition_interaction/summary.json")["adjusted_sensitivity"])
    for row in read("tutor_dependence/results.json")["results"]:
        put("gain_only." + row["section"] + ".equivalent", row["p"] < .05)
    for row in table("learning/adjusted_contrasts.csv").to_dict("records"):
        put("contrast." + row["scope"] + "." + row["contrast"], row)
    for row in read("learning/adjusted_tests.json"):
        put("learning_omnibus." + row["scope"], row)

    learning = read("learning/learning_figure_data.json")
    domains = learning["domains"]
    put("domains.count", len(domains))
    put("domains.distinct_winners", len({row["best_ai"]["arm_id"] for row in domains}))
    put("domains.ai_above_human", sum(row["best_ai"]["mean_gain_pp"] > row["human"]["mean_gain_pp"] for row in domains))
    put("domains.human_above_best_ai", sorted(row["topic"] for row in domains
        if row["human"]["mean_gain_pp"] > row["best_ai"]["mean_gain_pp"]))
    for row in domains:
        put("domain_winner." + row["topic"], row["best_ai"]["arm_id"])
        put("domain_provider." + row["topic"], row["best_ai"]["family"])
    for section in ["quant", "verbal"]:
        subset = [row for row in domains if row["section"] == section]
        put("domains." + section + ".best_ai_above_human", sum(
            row["best_ai"]["mean_gain_pp"] > row["human"]["mean_gain_pp"] for row in subset))
        put("domains." + section + ".human_above_pooled_ai", sum(
            row["human"]["mean_gain_pp"] > row["ai"]["mean_gain_pp"] for row in subset))
    put("irt", read("learning/irt_high_baseline_statistics.json"))
    irt = table("learning/irt_top_quartile_quant_arm_outcomes.csv").sort_values("descriptive_rank")
    put("irt.best_mean_gain", float(irt.iloc[0].mean_gain_pp))
    put("irt.top_five_gemini", int(irt.head(5).arm_id.str.startswith("gemini").sum()))
    quartiles = read("learning/quartile_thresholds.json")
    for scope, rows in quartiles.items():
        put("quartiles." + scope + ".cutpoints", [rows[k] for k in ["q25_points", "q50_points", "q75_points"]])
    for row in table("learning/proficiency_quartile_ai_control_contrasts.csv").to_dict("records"):
        if row["estimator"] == "ANCOVA HC3":
            put("quartile_contrast." + row["section"] + "." + row["quartile"], row)
    for row in read("learning/topic_proficiency_configuration_omnibus.json"):
        put("domain_interaction." + row["section"], row)
    cells = table("learning/topic_proficiency_configuration_cells.csv")
    put("domain_cells.significant", int((cells.ai_configuration_p_holm_across_28_cells < .05).sum()))
    quant = cells[cells.section == "quant"]
    put("domain_cells.quant_min_n", int(quant.minimum_ai_arm_n.min()))
    put("domain_cells.quant_max_n", int(quant.maximum_ai_arm_n.max()))
    put("domain_cells.quant_control_n", quant.groupby("quartile").n_control.first().tolist())
    put("domain_cells.effect_count", len(table("learning/topic_proficiency_configuration_effects.csv")))

    criteria = ["relevant_concepts", "concept_grouping", "concept_prioritization", "time_allocation",
                "practice_alignment", "difficulty_appropriateness", "answer_key_accuracy", "test_taking_strategies"]
    for name in ["planning_combined", "practice_combined", "common8_quant", "common8_verbal", "common8_combined", *criteria]:
        value = read("teaching/fit_" + name + ".json")
        if not value["converged"]:
            raise ValueError("Expert ranking did not converge: " + name)
        put("teaching." + name, {key: value[key] for key in ["n_decisive", "cluster_n", "n_all_grades"]})
        put("teaching." + name + ".winner", max(value["models"], key=lambda row: row["ability"])["model"])
        put("teaching." + name + ".models", len(value["models"]))
        put("teaching." + name + ".pairs", len(value["pairwise_contrasts"]))
        pairs = [(left, right) for index, left in enumerate(value["models"])
                 for right in value["models"][index + 1:]]
        separated = sum(left["hi"] < right["lo"] or right["hi"] < left["lo"] for left, right in pairs)
        put("teaching." + name + ".nonoverlapping_interval_fraction", separated / len(pairs))
    flags = table("teaching/flag_rates.csv")
    quant = flags[flags.scope == "quant"].sort_values("flags_per_100_answered", ascending=False)
    put("flags.quant_worst_model", quant.iloc[0].model_preset)
    put("flags.quant_worst_rate", float(quant.iloc[0].flags_per_100_answered))
    put("flags.verbal_max_rate", float(flags.loc[flags.scope == "verbal", "flags_per_100_answered"].max()))
    for row in read("teaching/flag_tests.json"):
        put("flags." + row["scope"] + ".p", row["p_holm_3"])
    for row in read("teaching/flag_expert_rank_agreement.json"):
        put("flag_agreement." + row["scope"] + "." + row["construct"], row)
    conversation = table("teaching/conversation_combined_leaderboard.csv")
    plotted = conversation[~conversation.arm.isin(["human", "gemini-3.6-flash-low"])]
    put("conversation.figure_ai_transcripts", int(plotted.n.sum()))
    ranges = [plotted.loc[plotted.arm.str.startswith(prefixes), "mean_strategy_score"]
              for prefixes in [("opus", "sonnet"), ("gpt",), ("gemini", "gemma"), ("kimi",)]]
    put("conversation.provider_bands_separate", all(
        left.max() < right.min() or right.max() < left.min()
        for index, left in enumerate(ranges) for right in ranges[index + 1:]))

    costs = table("costs/cost_efficiency_by_arm_scope.csv")
    latency = table("costs/latency_efficiency_by_arm_scope.csv")
    for row in costs.to_dict("records"):
        put("cost." + row["scope"] + "." + row["arm_id"], row)
    for scope in ["quant", "verbal", "combined"]:
        block = costs[(costs.scope == scope) & (costs.kind == "ai")]
        put("cost." + scope + ".cheapest_four", block.sort_values("cost_per_gain_pp").head(4).arm_id.tolist())
        put("cost." + scope + ".frontier", sorted(block.loc[block.pareto_frontier, "arm_id"].tolist()))
        reference = costs[(costs.scope == scope) & (costs.kind == "human")].iloc[0]
        put("learning." + scope + ".ai_means_above_human", int((block.mean_gain_pp > reference.mean_gain_pp).sum()))
    combined = costs[costs.scope == "combined"].set_index("arm_id")
    ai = combined[combined.kind == "ai"]
    minimum, maximum = float(ai.mean_cost_usd.min()), float(ai.mean_cost_usd.max())
    put("cost.minimum", minimum)
    put("cost.maximum", maximum)
    put("cost.frontier_maximum", float(ai.loc[ai.pareto_frontier, "mean_cost_usd"].max()))
    put("cost.ratio.human_gemma", float(combined.loc["human", "cost_per_gain_pp"] / combined.loc["gemma-4-31b-high", "cost_per_gain_pp"]))
    put("cost.ratio.pro_flash_session", float(combined.loc["gpt-5.5-pro-med", "mean_cost_usd"] / combined.loc["gemini-3.5-flash-low", "mean_cost_usd"]))
    lat = latency[latency.scope == "combined"].set_index("arm_id")
    put("latency.minimum", float(lat.median_latency_s.min()))
    put("latency.maximum", float(lat.median_latency_s.max()))
    put("latency.fastest", lat.median_latency_s.idxmin())
    put("latency.slowest", lat.median_latency_s.idxmax())
    put("cost.gemini31_dominates_gpt55pro", bool(
        combined.loc["gemini-3.1-pro-high", "mean_gain_pp"] > combined.loc["gpt-5.5-pro-med", "mean_gain_pp"]
        and combined.loc["gemini-3.1-pro-high", "mean_cost_usd"] < combined.loc["gpt-5.5-pro-med", "mean_cost_usd"]
        and lat.loc["gemini-3.1-pro-high", "median_latency_s"] < lat.loc["gpt-5.5-pro-med", "median_latency_s"]))
    for scope in ["quant", "verbal"]:
        block = costs[costs.scope == scope].set_index("arm_id")
        for arm in ["gemini-3.1-pro-high", "gemini-3.5-flash-low"]:
            put("cost." + scope + "." + arm + ".on_frontier", bool(block.loc[arm, "pareto_frontier"]))
        for left, right, name in [("gemini-3.5-flash-low", "gemini-3.6-flash-low", "flash35_cheaper_than36"),
                                  ("opus-5-high", "opus-4.8-xhigh", "opus5_cheaper_than48")]:
            put("cost." + scope + "." + name,
                bool(block.loc[left, "cost_per_gain_pp"] < block.loc[right, "cost_per_gain_pp"]))
    put("resources", read("costs/statistics.json"))
    rows = [json.loads(line) for line in (root / "costs/individual_model_equivalence.jsonl").open()]
    sources["costs/individual_model_equivalence.jsonl"] = sha256(root / "costs/individual_model_equivalence.jsonl")
    individual = [row for row in rows if row["record_kind"] == "model_result"]
    put("individual.passing", sorted(row["arm_id"] for row in individual if row["nominal_equivalent"]))
    for row in individual:
        put("individual." + row["arm_id"], row)
    put("individual.summary", next(row for row in rows if row["record_kind"] == "summary"))
    if data_dir is not None:
        requests = observed = 0
        for section in ["quant", "verbal"]:
            path = Path(data_dir) / "3_cost_data" / ("gre_" + section) / "llm_cost_records.csv"
            sources["dataset/3_cost_data/gre_" + section + "/llm_cost_records.csv"] = sha256(path)
            for chunk in pd.read_csv(path, usecols=["token_usage_recorded"], chunksize=20000):
                requests += len(chunk)
                observed += int(chunk.token_usage_recorded.astype(str).str.lower().isin(["true", "1"]).sum())
        put("requests.total", requests)
        put("requests.observed", observed)
        put("requests.covered_percent", 100 * observed / requests)
        put("requests.unpriced_percent", 100 * (requests - observed) / requests)

    engagement = read("engagement/results.json")
    put("engagement.sessions", engagement["sessions"])
    put("engagement.temporal_sessions", engagement["temporal_sessions"])
    put("engagement.test_count", len(engagement["tests"]))
    put("engagement.family_counts", dict(Counter(row["family"] for row in engagement["tests"])))
    for row in engagement["selected_figure"]:
        put("engagement." + row["scope"] + "." + row["edge"], row)
    selected_temporal = {
        "quant_early_engagement_late_practice": "temporal:quant:student_chat_messages_first30:practice_first_credit_closed_minutes30to60",
        "verbal_late_practice_gain": "temporal:verbal:practice_first_credit_closed_minutes30to60:gain_pp"}
    by_id = {row["record_id"]: row for row in engagement["tests"]}
    for name, identity in selected_temporal.items():
        put("engagement.temporal." + name, by_id[identity]["p_holm276"])
    robust = read("engagement_robustness/results.json")
    put("engagement.repeat_signs_preserved", robust["all_repeat_signs_preserved"])
    put("engagement.repeat_selected", len(robust["repeat_participation"]))
    put("engagement.repeat_sessions", robust["repeat_participation"][0]["exclude_repeat"]["n"])
    selected_pairs = [("latency_mean_s", "student_chat_messages"),
                      ("student_chat_messages", "practice_first_credit_closed"),
                      ("practice_first_credit_closed", "gain_pp")]
    for scope in ["quant", "combined"]:
        directions = []
        for x, y in selected_pairs:
            baseline = by_id[f"primary_raw:{scope}:{x}:{y}"]["estimate_per_unit"]
            log = by_id[f"secondary_log1p:{scope}:{x}:{y}"]["estimate_per_unit"]
            directions.append(np.sign(log) == np.sign(baseline))
            for row in robust["diagnostic_fits"]:
                if (row["scope"], row["x"], row["y"]) == (scope, x, y):
                    estimate = row.get("estimate_per_unit", row.get("partial_rank_correlation"))
                    directions.append(np.sign(estimate) == np.sign(baseline))
        put("engagement." + scope + ".robust_directions_preserved", bool(all(directions)))
    repeats = read("repeat_main/results.json")
    put("repeat", repeats)
    for label, analysis in repeats["analyses"].items():
        put("repeat." + label + ".passing", sorted(row["arm_id"] for row in analysis["individual_combined_equivalence"] if row["p"] < .05))
        for kind in ["ai", "human", "control"]:
            put("repeat." + label + "." + kind, sum(row["n"] for row in analysis["by_section_kind"] if row["kind"] == kind))
        for row in analysis["adjusted_contrasts"]:
            if row["scope"] == "combined" and row["contrast"] == "pooled_ai_minus_control":
                put("repeat." + label + ".ai_control", row["estimate_pp"])
    put("separation", read("leaderboards/summary.json")["common_comparison"])

    reviewer = read("reviewer/results.json")
    for name, block in reviewer["results"].items():
        for method in ["original", "sensitivity"]:
            fit = block[method]
            put("reviewer." + name + "." + method + ".df", fit["df"])
            for row in fit["pairwise_contrasts"]:
                put("reviewer." + name + "." + method + "." + row["model_a"] + ":" + row["model_b"], row)

    prompts = read("prompts/summary.json")
    put("prompts.pilot_sessions", prompts["pilot_sessions"])
    for row in prompts["coverage"]:
        put("prompts." + row["dataset"] + ".coverage", row)
    for row in prompts["fits"]:
        put("prompts." + row["record_id"], {k: row[k] for k in ["n", "n_clusters", "interaction_df", "interaction_chi2", "interaction_p"]})
    for row in prompts["contrasts"]:
        put("prompts." + row["record_id"], row)
    put("prompts.pilot.significant_contrasts", sum(row["p_holm"] < .05
        for row in prompts["contrasts"] if row["dataset"] == "pilot" and row["covariance"] == "cluster"))
    groups = table("prompts/group_estimates.csv")
    for row in groups.to_dict("records"):
        put("prompts." + row["record_id"], row)
    pilot = groups[(groups.dataset == "pilot") & groups.model.notna()]
    put("prompts.pilot.models", int(pilot.model.nunique()))
    put("prompts.pilot.conditions", int(groups.loc[groups.dataset == "pilot", "condition"].nunique()))
    comparisons = []
    for _, block in pilot.groupby("model"):
        means = block.set_index("condition").mean_gain_pp
        comparisons.append(means["lean/lean"] > means["max/max"])
    put("prompts.pilot.minimal_gains_higher_each_model", bool(all(comparisons)))
    return metrics, sources


def compare(actual, rule):
    """Apply exactly the printed precision or inequality, never a loose tolerance."""
    operation, expected = rule["operation"], rule["expected"]
    if operation == "equal":
        return plain(actual) == expected
    values = np.asarray(actual, float)
    if not np.isfinite(values).all():
        return False
    if operation == "round":
        rounded = np.round(values, rule["digits"])
        target = np.asarray(expected, float)
        return bool(rounded.shape == target.shape and np.array_equal(rounded, target))
    if operation == "less_than":
        return bool(np.all(values < expected))
    if operation == "greater_than":
        return bool(np.all(values > expected))
    if operation == "close":
        return bool(np.allclose(values, expected, rtol=0, atol=rule["absolute_tolerance"]))
    raise ValueError("Unknown comparison rule: " + operation)


def verify_metrics(metrics, rules, output_dir, signature):
    """Save each check before raising; interrupted checks resume by fingerprint."""
    journal = Journal(Path(output_dir) / "checks.jsonl")
    checks = []
    seen = set()
    for rule in rules:
        identity = rule["id"]
        if identity in seen:
            raise ValueError("Duplicate paper check: " + identity)
        seen.add(identity)
        metric = rule["metric"]
        value = metrics.get(metric)
        local = hashlib.sha256((signature + json.dumps([rule, value], sort_keys=True, allow_nan=False)).encode()).hexdigest()
        check = journal.get(identity, local)
        if check is None:
            check = journal.save(identity, local, dict(id=identity, metric=metric,
                locations=rule["locations"], actual=value, expected=rule["expected"],
                operation=rule["operation"], passed=metric in metrics and compare(value, rule)))
        checks.append(check)
    failures = [row for row in checks if not row["passed"]]
    result = dict(complete=True, passed=not failures, checks=len(checks),
                  failures=failures, comparison="Printed precision, stated inequalities and exact rankings")
    write_json(Path(output_dir) / "summary.json", result)
    if failures:
        raise ValueError("Paper prose checks failed: " + ", ".join(row["id"] for row in failures))
    return result


def run(analysis_dir, output_dir, data_dir=None):
    metrics, sources = collect_metrics(analysis_dir, data_dir)
    expected = json.loads(EXPECTED.read_text())
    signature = sha256(EXPECTED) + sha256(Path(__file__)) + json.dumps(sources, sort_keys=True)
    write_json(Path(output_dir) / "sources.json", sources)
    write_json(Path(output_dir) / "metrics.json", {rule["metric"]: metrics.get(rule["metric"]) for rule in expected["checks"]})
    return verify_metrics(metrics, expected["checks"], output_dir, signature)
