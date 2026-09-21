# Figures, in paper order

Current paper numbers map to stable script filenames below. Each statistical figure has a separate plotting script. Run the analyses first, then draw a figure:

```bash
python figures/figure_01_learning.py --analysis-root results/analysis --output results/figures
```

Every script saves PDF, SVG and PNG. Install Arial to match the paper's fonts, or use an available font. Add `--native-figures` when using modified data or changing the layout. The study-design diagram and screenshots are included as image files.

| Figure | Paper content | Code or asset |
| --- | --- | --- |
| 1 | Learning gains | [figure_01_learning.py](figure_01_learning.py) |
| 2 | Study design | [figure_02_study_design.pdf](../assets/static/figure_02_study_design.pdf) |
| 3 | Teaching evaluations | [figure_03_teaching_evaluations.py](figure_03_teaching_evaluations.py) |
| 4 | Cost, reply time and engagement | [figure_04_resources.py](figure_04_resources.py) |
| 5 | Cost per learning gain | [figure_05_cost_per_gain.py](figure_05_cost_per_gain.py) |
| 6 | Reply time, student engagement and first-attempt correct practice | [figure_06_engagement_practice.py](figure_06_engagement_practice.py) |
| C.1 | AI tutoring interface | [figure_07_tutoring_interface.png](../assets/static/figure_07_tutoring_interface.png) |
| C.2 | Assessment interface | [figure_08_assessment_interface.png](../assets/static/figure_08_assessment_interface.png) |
| C.3 | Human tutor pre-test review | [figure_09_human_pretest.pdf](../assets/static/figure_09_human_pretest.pdf) |
| C.4 | Human tutoring video call | [figure_10_human_tutoring.pdf](../assets/static/figure_10_human_tutoring.pdf) |
| C.5 | Expert comparison interface | [figure_11_expert_comparison_interface.png](../assets/static/figure_11_expert_comparison_interface.png) |
| E.1 | Observed and adjusted section leaderboards | [figure_12_section_learning.py](figure_12_section_learning.py) |
| E.2 | Seven domain leaderboards | [figure_13_domain_learning.py](figure_13_domain_learning.py) |
| E.3 | Starting proficiency contrasts | [figure_14_starting_proficiency.py](figure_14_starting_proficiency.py) |
| E.4 | Quantitative domain and proficiency | [figure_15_domain_proficiency.py](figure_15_domain_proficiency.py) |
| F.1 | Section cost per gain | [figure_16_section_cost_per_gain.py](figure_16_section_cost_per_gain.py) |
| F.2 | Section resource frontiers | [figure_17_section_frontiers.py](figure_17_section_frontiers.py) |
| F.3 | Section reply time and engagement | [figure_18_reply_time_engagement.py](figure_18_reply_time_engagement.py) |
| F.4 | Tutoring experience | [figure_19_tutoring_experience.py](figure_19_tutoring_experience.py) |
| G.1 | Criteria and overall expert preference | [figure_20_expert_criteria.py](figure_20_expert_criteria.py) |
| H.1 | Combined conversation indicators | [figure_21_conversation_indicators.py](figure_21_conversation_indicators.py) |
| H.2 | Section conversation indicators | [figure_22_conversation_by_section.py](figure_22_conversation_by_section.py) |
| H.3 | Student disputed answers | [figure_23_disputed_answers.py](figure_23_disputed_answers.py) |
| I.1 | Minimal versus expanded prompts | [figure_24_prompt_comparisons.py](figure_24_prompt_comparisons.py) |
