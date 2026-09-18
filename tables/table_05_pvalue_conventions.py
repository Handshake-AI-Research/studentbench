"""Table 5: fixed study protocol definitions, not estimated outcomes."""

from studentbench.table_output import render, cli

ROWS = [
    {
        "Analysis": "Equivalence and its sensitivity checks",
        "Reported p and comparison family": "Per-comparison TOST. Additional Holm checks cover "
        "twelve individual AI tutors, 329 subgroup variants "
        "(excluding the original pooled test), and two margins "
        "for the subset of five AI tutors, as separate "
        "families.",
    },
    {
        "Analysis": "Tutoring versus no tutoring",
        "Reported p and comparison family": "Per-comparison tests for the six section-specific and "
        "combined contrasts in Table 6.",
    },
    {
        "Analysis": "Exploratory IRT subgroup",
        "Reported p and comparison family": "Per-comparison test for the specified "
        "highest-proficiency quartile; the full sensitivity "
        "family contains 24 specifications per GRE scope.",
    },
    {
        "Analysis": "Learning differences among AI tutors",
        "Reported p and comparison family": "Holm across Quantitative, Verbal and Combined, "
        "separately for observed-gain and adjusted-gain omnibus "
        "tests.",
    },
    {
        "Analysis": "Human–AI gap between sections",
        "Reported p and comparison family": "Per-comparison tests; a separate Holm check covers "
        "both the unadjusted and baseline/form-adjusted tests.",
    },
    {
        "Analysis": "Starting-proficiency groups",
        "Reported p and comparison family": "Holm across eight section–quartile comparisons, "
        "separately for observed and adjusted gains.",
    },
    {
        "Analysis": "AI tutor, domain and proficiency interaction",
        "Reported p and comparison family": "Holm across two sections; separate Holm families for "
        "28 domain–quartile omnibus tests and 364 contrasts by "
        "AI tutor, domain and quartile.",
    },
    {
        "Analysis": "Tutoring contrasts within domains",
        "Reported p and comparison family": "Per-comparison p; q denotes Benjamini–Hochberg "
        "correction across four Quantitative or three Verbal "
        "domains, separately for AI and human tutoring.",
    },
    {
        "Analysis": "Expert preferences",
        "Reported p and comparison family": "Holm across all pairs of AI tutor entries within each "
        "ranking: 78 for 13 version-specific entries or 66 for "
        "12 entries.",
    },
    {
        "Analysis": "Student answer disputes",
        "Reported p and comparison family": "Holm across three scope-level tests; expert-ranking "
        "correlations use a separate six-test family.",
    },
    {
        "Analysis": "Reply time and student messages",
        "Reported p and comparison family": "Holm across six correlations: Pearson and Spearman in "
        "each of three scopes.",
    },
    {
        "Analysis": "AI tutor cost, reply time, messages and gain",
        "Reported p and comparison family": "Per-comparison exploratory correlations; an additional "
        "Holm check covers all 18 tests (three predictors, two "
        "correlation methods and three scopes).",
    },
    {
        "Analysis": "Dialogue, practice and learning",
        "Reported p and comparison family": "Holm across 276 exploratory tests for Figure 6. "
        "Earlier analyses retain separate families of eight "
        "original session regressions, six additional session "
        "regressions, twelve correlations across AI tutors and "
        "eight student-message regressions.",
    },
    {
        "Analysis": "Interaction measures",
        "Reported p and comparison family": "Holm across 21 omnibus tests; pairwise tests use "
        "separate families of 66 or 78 pairs of AI tutors for "
        "each measure and scope.",
    },
    {
        "Analysis": "Prompt comparisons",
        "Reported p and comparison family": "Per-comparison pilot omnibus tests; Holm across six "
        "pilot contrasts, and separately across three "
        "later-cohort contrasts.",
    },
]


def run(data_dir, analysis_dir, output_dir):
    return render(
        output_dir,
        "table_05_pvalue_conventions",
        ROWS,
        "Table 5. P-value reporting rules",
        "Fixed study protocol; these definitions specify the analyses rather than report fitted results.",
    )


if __name__ == "__main__":
    cli(run)
