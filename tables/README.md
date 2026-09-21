# Tables in the paper

Current paper numbers map to stable script filenames below. Each script writes CSV, Markdown and LaTeX. Four tables contain study definitions; the rest are calculated from the analysis outputs.

| Table | Code |
|---|---|
| Table C.1: starting scores and assessment-form order | [table_01_baseline.py](table_01_baseline.py) |
| Table C.2: primary exclusion counts at the two screening stages | [table_02_data_quality.py](table_02_data_quality.py) |
| Table C.3: recorded tutor endpoints and analyzed assignments | [table_03_tutor_configurations.py](table_03_tutor_configurations.py) |
| Table C.4: prompt guidance and scaffolding | [table_04_prompt_guidance.py](table_04_prompt_guidance.py) |
| Table D.1: p-value conventions | [table_05_pvalue_conventions.py](table_05_pvalue_conventions.py) |
| Table D.2: adjusted tutoring-minus-no-tutor learning contrasts | [table_06_adjusted_contrasts.py](table_06_adjusted_contrasts.py) |
| Table D.3: equivalence with CR2 uncertainty for shared human tutors | [table_07_tutor_dependence.py](table_07_tutor_dependence.py) |
| Table D.4: equivalence and engagement results after excluding repeat participants | [table_08_repeat_participation.py](table_08_repeat_participation.py) |
| Table E.1: observed and adjusted Combined learning gains | [table_09_combined_learning.py](table_09_combined_learning.py) |
| Table E.2: seven domain contrasts against no tutoring | [table_10_domain_contrasts.py](table_10_domain_contrasts.py) |
| Table F.1: correlations of tutor-level resources with learning gain | [table_11_resource_gain_correlations.py](table_11_resource_gain_correlations.py) |
| Table F.2: selected adjusted engagement, correct-practice, and gain associations | [table_12_engagement_practice.py](table_12_engagement_practice.py) |
| Table G.1: lesson-planning and practice-problem rubric criteria | [table_13_rubric_criteria.py](table_13_rubric_criteria.py) |
| Table G.2: expert rankings with reviewer and pre-test clustering | [table_14_reviewer_dependence.py](table_14_reviewer_dependence.py) |
| Table H.1: conversation-indicator rules | [table_15_conversation_rules.py](table_15_conversation_rules.py) |
| Table I.1: all five exploratory prompt-pilot settings | [table_16_prompt_pilot.py](table_16_prompt_pilot.py) |

Run the analyses and tables together:

```bash
python reproduce.py --data ../data --output results --verify
```

Or render a single table from existing analysis outputs:

```bash
python -m tables.table_09_combined_learning --data ../data --analysis-root results/analysis --output results/tables
```
