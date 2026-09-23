"""Table F.1: fixed study protocol definitions, not estimated outcomes."""

from tables.output import render
from studentbench.table_output import cli

ROWS = [
    {
        "Indicator": "Scaffolding cues",
        "Detection rule": "A tutor response following a student message of at least two words "
        "contains evaluative language, or shares at least two substantive "
        "words with that message and contains a scaffolding cue.",
    },
    {
        "Indicator": "Request for explanation",
        "Detection rule": "At least one tutor question contains a request for reasoning or "
        "explanation.",
    },
    {
        "Indicator": "Interactive questions",
        "Detection rule": "At least four tutor messages contain a question mark or an "
        "invitation to respond.",
    },
    {
        "Indicator": "Early attempt request",
        "Detection rule": "At least one of the first five tutor messages asks the student to "
        "try or attempt a problem.",
    },
    {
        "Indicator": "Long solution after a reply",
        "Detection rule": "After a student message of at least two words, a tutor message of at "
        "least 140 words includes a solution or calculation cue.",
    },
    {
        "Indicator": "Reasoning checks",
        "Detection rule": "At least two tutor questions contain a reasoning-check or strategy "
        "cue.",
    },
]


def run(data_dir, analysis_dir, output_dir):
    return render(
        output_dir,
        "table_15_conversation_rules",
        ROWS,
        "Table F.1. Conversation indicator definitions",
        "Fixed study protocol; these definitions specify the analyses rather than report fitted results.",
    )


if __name__ == "__main__":
    cli(run)
