# Checks of the paper's reported results

`studentbench/paper_results.py` checks the empirical numbers and rankings stated
in the body and appendix of paper version `53f1608`. The comparison targets are
in `verification/paper_results_expected.json`; each names the metric, source
location, and printed precision or inequality. Repeated statements share a check.

The checks read fresh analysis outputs. Expected values never enter feature
extraction, model fitting or rendering. Each completed check is saved separately;
changed inputs invalidate saved passes, and any mismatch stops verification.

| Claims | Recomputed outputs |
| --- | --- |
| Participants, ages, experts, reviews, transcripts and interaction counts | `study_summary` |
| Pooled and section-specific equivalence, including repeat exclusion | `primary_equivalence` |
| Control contrasts, domain leaders, proficiency groups and IRT findings | `learning` |
| Expert-ranking winners and answer-dispute associations | `teaching`, `reviewer` |
| Cost and reply-time ranges, Pareto frontiers and individual tutor equivalence | `costs` |
| Dialogue, practice and learning associations and robustness | `engagement`, `engagement_robustness` |
| Repeat-participation conclusions | `repeat_main` |
| Separation across five leaderboards | `leaderboards` |
| Pilot and later-cohort prompt comparisons | `prompts` |

Full-precision estimator checks, table-cell checks and figure checks are separate.
This module checks the presentation of those estimates: for example, whether a
probability rounds to `.015`, an interval rounds to `[-2.18, 1.03]`, or all named
tutors belong to the passing set.

The human/Gemma cost-per-gain ratio rounds to `918×`; after removing repeat
participants it rounds to `932×`. Individual equivalence uses the same adjusted
CR2 model as the pooled comparison. Six tutors pass in the full cohort; these six
and GPT-5.4 mini pass after repeat-participant exclusion.

Protocol settings, recruitment descriptions and facts attributed to outside
sources are not participant-result estimates. Examples include the invitation
count, compensation, assessment time limits, bootstrap settings, and cited
benchmark scores. They are not labeled as statistically reproduced by these
checks. The released protocol parameters, questionnaires and analysis code
document the study settings; external-source claims require their cited sources.

Pooled demographic values use the released population aggregates. Combined CR2
uncertainty uses the global moment sums described in
[`assets/analysis/README.md`](../assets/analysis/README.md), accounting for repeated participants and tutors across sections.

The leave-one-human-tutor-out sensitivity uses anonymous tutor-alias groups and
recomputes all 13 omitted-cohort regressions. As for other combined CR2 fits,
global moment inputs preserve the contribution of repeated students. Direct
identity mappings and participant-pair lists are not included. The published
cluster-count metadata describes the source linkage calculation.
