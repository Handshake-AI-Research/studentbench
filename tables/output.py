"""Write readable tables and the exact LaTeX table fragments used by the paper.

Templates contain formatting and study definitions. Every empirical body value
is inserted from the freshly calculated rows, never from verification targets.
"""

from pathlib import Path
import json
import re

from studentbench.table_output import render as render_readable

NUMBER = re.compile(r"(?<![A-Za-z])[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def cell_numbers(value):
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", str(value)).replace("−", "-")
    return [float(match.group()) for match in NUMBER.finditer(value)]


def render(output_dir, name, rows, title, source, *, values=None):
    result = render_readable(output_dir, name, rows, title, source)
    templates = Path(__file__).with_name("templates")
    specification = json.loads((templates / (name + ".json")).read_text())
    text = (templates / (name + ".tex")).read_text()
    fields = list(rows[0])
    for binding in specification["bindings"]:
        value = cell_numbers(rows[binding["row"]][fields[binding["column"]]])[
            binding["number"]
        ]
        formatted = format(value, binding["format"])
        if binding.get("omit_leading_zero", False):
            formatted = formatted.removeprefix("0")
        placeholder = f"@@R{binding['row']}C{binding['column']}N{binding['number']}@@"
        text = text.replace(placeholder, formatted)
    for key, value in (values or {}).items():
        formatted = value if isinstance(value, str) else f"{int(value):,}"
        text = text.replace("@@" + key + "@@", formatted)
    if "@@" in text:
        raise ValueError(f"Unresolved table placeholder: {name}")
    target = Path(output_dir) / (name + "_paper.tex")
    target.write_text(text)
    return result
