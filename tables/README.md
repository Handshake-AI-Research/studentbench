# Tables in the paper

Each script writes CSV, Markdown and LaTeX. Empirical tables read fresh analysis outputs; fixed protocol tables are identified in their source descriptions.

| Table | Code |
|---|---|
| Table 1: starting scores and assessment-form order | [table_01_baseline.py](table_01_baseline.py) |
| Table 2: primary exclusion counts at the two screening stages | [table_02_data_quality.py](table_02_data_quality.py) |
| Table 3: recorded tutor endpoints and analyzed assignments | [table_03_tutor_configurations.py](table_03_tutor_configurations.py) |
| Table 4: prompt guidance and scaffolding | [table_04_prompt_guidance.py](table_04_prompt_guidance.py) |
| Table 5: p-value conventions | [table_05_pvalue_conventions.py](table_05_pvalue_conventions.py) |
| Table 6: adjusted tutoring-minus-no-tutor learning contrasts | [table_06_adjusted_contrasts.py](table_06_adjusted_contrasts.py) |
| Table 7: equivalence with CR2 uncertainty for shared human tutors | [table_07_tutor_dependence.py](table_07_tutor_dependence.py) |
| Table 8: equivalence and engagement results after excluding repeat participants | [table_08_repeat_participation.py](table_08_repeat_participation.py) |
| Table 9: observed and adjusted Combined learning gains | [table_09_combined_learning.py](table_09_combined_learning.py) |
| Table 10: seven domain contrasts against no tutoring | [table_10_domain_contrasts.py](table_10_domain_contrasts.py) |
| Table 11: correlations of tutor-level resources with learning gain | [table_11_resource_gain_correlations.py](table_11_resource_gain_correlations.py) |
| Table 12: selected adjusted engagement, correct-practice, and gain associations | [table_12_engagement_practice.py](table_12_engagement_practice.py) |
| Table 13: lesson-planning and practice-problem rubric criteria | [table_13_rubric_criteria.py](table_13_rubric_criteria.py) |
| Table 14: expert rankings with reviewer and pre-test clustering | [table_14_reviewer_dependence.py](table_14_reviewer_dependence.py) |
| Table 15: conversation-indicator rules | [table_15_conversation_rules.py](table_15_conversation_rules.py) |
| Table 16: all five exploratory prompt-pilot settings | [table_16_prompt_pilot.py](table_16_prompt_pilot.py) |

Run the analyses and tables together:

```bash
python reproduce.py --data data --output results --verify
```

Or render a single table from existing analysis outputs:

```bash
python -m tables.table_09_combined_learning --data data --analysis-root results/analysis --output results/tables
```
