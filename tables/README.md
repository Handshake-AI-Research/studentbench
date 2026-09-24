# Tables in the paper

Each script computes one table from the released observations or the analysis outputs. It writes CSV, Markdown, a plain LaTeX table, and a `_paper.tex` fragment with the paper's formatting. Four tables describe the study protocol; the reporting-rules table also counts the tutors that pass the individual equivalence tests.

| Paper table | Code |
|---|---|
| A.1: Starting scores and assessment-form order | [table_01_baseline.py](table_01_baseline.py) |
| A.2: Data-quality exclusions | [table_02_data_quality.py](table_02_data_quality.py) |
| A.3: AI tutor endpoints and sample sizes | [table_03_tutor_configurations.py](table_03_tutor_configurations.py) |
| A.4: Prompt guidance | [table_04_prompt_guidance.py](table_04_prompt_guidance.py) |
| B.1: Statistical reporting rules | [table_05_pvalue_conventions.py](table_05_pvalue_conventions.py) |
| B.2: Tutoring gains relative to control | [table_06_adjusted_contrasts.py](table_06_adjusted_contrasts.py) |
| B.3: Results after excluding repeat participants | [table_08_repeat_participation.py](table_08_repeat_participation.py) |
| C.1: Combined learning gains by tutor | [table_09_combined_learning.py](table_09_combined_learning.py) |
| C.2: Domain gains relative to control | [table_10_domain_contrasts.py](table_10_domain_contrasts.py) |
| D.1: Resources and learning gain across AI tutors | [table_11_resource_gain_correlations.py](table_11_resource_gain_correlations.py) |
| D.2: Engagement, correct practice and learning | [table_12_engagement_practice.py](table_12_engagement_practice.py) |
| E.1: Expert rubric criteria | [table_13_rubric_criteria.py](table_13_rubric_criteria.py) |
| E.2: Expert rankings with repeated-reviewer clustering | [table_14_reviewer_dependence.py](table_14_reviewer_dependence.py) |
| F.1: Conversation-indicator rules | [table_15_conversation_rules.py](table_15_conversation_rules.py) |
| G.1: Prompt-pilot settings | [table_16_prompt_pilot.py](table_16_prompt_pilot.py) |

Run all analyses and tables:

```bash
python reproduce.py --data ../data --output results --verify
```

Or render one table from existing analysis outputs:

```bash
python -m tables.table_09_combined_learning --data ../data --analysis-root results/analysis --output results/tables
```

The files in `templates/` preserve the paper's table layout, labels and protocol descriptions. Numeric placeholders receive the newly computed results. Verification checks every table cell and compares each `_paper.tex` fragment byte for byte with the published source. These fragments use the paper's LaTeX preamble and cross-references; they are not standalone documents. Table C.1 uses the four provider logos in `assets/brandmarks/`.
