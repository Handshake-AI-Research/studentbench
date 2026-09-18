"""Table 4: fixed study protocol definitions, not estimated outcomes."""

from studentbench.table_output import render, cli

ROWS = [
    {
        "Component": "GRE context",
        "Main-study system templates": "Brief content and format overview",
        "Additional guidance in the detailed variants": "Detailed test structure, pacing, scoring "
        "conventions and format-specific "
        "strategies",
    },
    {
        "Component": "Diagnosis and sequencing",
        "Main-study system templates": "AI tutor identifies needs and orders concepts",
        "Additional guidance in the detailed variants": "Explicit question-by-question "
        "misconception diagnosis, response-time "
        "interpretations and prerequisite "
        "sequencing",
    },
    {
        "Component": "Difficulty calibration",
        "Main-study system templates": "AI tutor selects appropriate concepts and practice",
        "Additional guidance in the detailed variants": "Instructions for interpreting easy versus "
        "hard mistakes and using supplied expert "
        "ratings",
    },
    {
        "Component": "Practice progression",
        "Main-study system templates": "Generate practice with worked solutions",
        "Additional guidance in the detailed variants": "Vary surface forms and progress from "
        "easier to difficult GRE-level practice",
    },
    {
        "Component": "Conversational teaching",
        "Main-study system templates": "Teach transferable methods; choose explanation and practice",
        "Additional guidance in the detailed variants": "Attempt before hints, guide students to "
        "identify mistakes, request explanations "
        "and predictions, and connect conceptual "
        "understanding to efficient GRE methods",
    },
]


def run(data_dir, analysis_dir, output_dir):
    return render(
        output_dir,
        "table_04_prompt_guidance",
        ROWS,
        "Table 4. Omitted prompt guidance",
        "Fixed study protocol; these definitions specify the analyses rather than report fitted results.",
    )


if __name__ == "__main__":
    cli(run)
