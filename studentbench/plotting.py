"""Shared labels, colors, output formats and command-line options for paper figures."""

from pathlib import Path
import argparse
import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

COLORS = {
    "OpenAI": "#0072B2",
    "Anthropic": "#D55E00",
    "Google": "#009E73",
    "Moonshot": "#A96F00",
    "human": "#b33a2b",
    "control": "#86939d",
}
HUMAN_RED = COLORS["human"]
PARETO_BLUE = "#087ca7"
PARETO_SLATE = "#86939d"
PARETO_TEXT = MODEL_TEXT = "#4b5964"
AXIS_GRAY = "#aeb8bf"
GRID_GRAY = "#e6eaed"
TICK_TEXT = "#38444c"
INTERVAL_GRAY = "#86939d"
SECTION_COLORS = {"quant": "#7B3294", "verbal": "#E6D34A"}
SECTION_LINE_COLORS = {"quant": "#7B3294", "verbal": "#9A8200"}
SECTION_TEXT_COLORS = {"quant": "#7B3294", "verbal": "#806D00"}
SECTION_NAMES = {"quant": "Quantitative", "verbal": "Verbal", "combined": "Combined"}
DOMAIN_ORDER = (
    ("quant", "Data Analysis", 7),
    ("quant", "Geometry", 5),
    ("quant", "Arithmetic", 8),
    ("quant", "Algebra", 7),
    ("verbal", "Sentence Equivalence", 7),
    ("verbal", "Text Completion", 7),
    ("verbal", "Reading Comprehension", 13),
)
QUARTILE_IDS = ("Q1", "Q2", "Q3", "Q4")
MODEL_LABELS = {
    "gemini-3.1-pro-high": "Gemini 3.1 Pro (high)",
    "gemini-3.5-flash-low": "Gemini 3.5 Flash (low)",
    "gemini-3.6-flash-low": "Gemini 3.6 Flash (low)",
    "gemini-3.7-flash-medium": "Gemini 3.7 Flash (med)",
    "gemma-4-31b-high": "Gemma 4 31B (high)",
    "gpt-5.4-mini-none": "GPT-5.4 mini (off)",
    "gpt-5.5-high": "GPT-5.5 (high)",
    "gpt-5.5-pro-med": "GPT-5.5 Pro (med)",
    "kimi-k2.6": "Kimi K2.6 (med)",
    "opus-4.8-off": "Opus 4.8 (off)",
    "opus-4.8-xhigh": "Opus 4.8 (x-high)",
    "opus-5-high": "Opus 5 (high)",
    "sonnet-4.6-low": "Sonnet 4.6 (low)",
    "sonnet-5-low": "Sonnet 5 (low)",
    "sonnet-section-specific-low": "Sonnet (section-specific)",
    "human": "Human tutor",
    "control": "No tutor",
}


def display_label(value):
    label = MODEL_LABELS.get(str(value), str(value))
    label = label.replace("(medium)", "(med)").replace("(xhigh)", "(x-high)")
    label = label.replace("GPT-5.4 mini (none)", "GPT-5.4 mini (off)")
    return re.sub(r"\bKimi K2\.6\b(?!\s*\(med\))", "Kimi K2.6 (med)", label)


def model_family(arm):
    for prefixes, family in [
        (("gpt",), "OpenAI"),
        (("gemini", "gemma"), "Google"),
        (("opus", "sonnet"), "Anthropic"),
        (("kimi",), "Moonshot"),
    ]:
        if str(arm).startswith(prefixes):
            return family
    return str(arm)


def model_color(arm):
    return COLORS.get(model_family(arm), "#647687")


def configure(profile="learning"):
    """Use each original figure family's settings; reset between figure renders."""
    plt.rcdefaults()
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = "Arial" if "Arial" in available else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 12,
            "mathtext.fontset": "custom",
            "mathtext.rm": font,
            "mathtext.it": font + ":italic",
            "mathtext.bf": font + ":bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    if profile == "learning":
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.titlesize": 13,
                "axes.titleweight": "bold",
                "axes.labelsize": 11,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.spines.left": False,
                "axes.edgecolor": AXIS_GRAY,
                "axes.linewidth": 0.7,
                "xtick.color": TICK_TEXT,
                "ytick.color": TICK_TEXT,
                "axes.labelcolor": "black",
                "axes.titlecolor": "black",
            }
        )
    elif profile in {"teaching", "criteria"}:
        plt.rcParams.update(
            {
                "font.size": 8 if profile == "teaching" else 6.75081982 / (482.4 / 756),
                "axes.labelcolor": "black",
                "axes.titlecolor": "black",
                "text.color": "black",
                "axes.edgecolor": AXIS_GRAY,
                "axes.linewidth": 0.7,
                "xtick.color": TICK_TEXT,
                "ytick.color": MODEL_TEXT,
            }
        )
    elif profile in {"efficiency", "cost"}:
        plt.rcParams.update(
            {
                "font.size": 11 if profile == "efficiency" else 9,
                "text.color": "black" if profile == "efficiency" else MODEL_TEXT,
                "axes.labelcolor": "black",
                "xtick.color": TICK_TEXT,
                "ytick.color": TICK_TEXT,
            }
        )
    elif profile == "prompts":
        plt.rcParams.update(
            {
                "font.size": 9,
                "axes.titlesize": 10,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 9,
            }
        )
    elif profile != "main":
        raise ValueError(f"Unknown figure style: {profile}")


def save_figure(
    figure,
    output_dir,
    name,
    *,
    bbox_inches="tight",
    pad_inches=0.1,
    extra_canvas=(0, 0),
):
    """Save each completed figure immediately in editable and ready-to-use formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The final manuscript reserves this small extra border for authored titles.
    original_bounds = figure.get_tightbbox
    if any(extra_canvas):
        from matplotlib.transforms import Bbox

        def bounds_with_margin(*args, **kwargs):
            box = original_bounds(*args, **kwargs)
            return Bbox.from_extents(
                box.x0,
                box.y0,
                box.x1 + extra_canvas[0] / 72,
                box.y1 + extra_canvas[1] / 72,
            )

        figure.get_tightbbox = bounds_with_margin
    try:
        for extension in ("pdf", "svg", "png"):
            options = {"dpi": 160} if extension == "png" else {}
            figure.savefig(
                output_dir / f"{name}.{extension}",
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                **options,
            )
    finally:
        figure.get_tightbbox = original_bounds
    plt.close(figure)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=lambda value: value.item()) + "\n"
    )


def figure_cli(render):
    parser = argparse.ArgumentParser(description=render.__doc__)
    parser.add_argument("--analysis-root", type=Path, default=Path("results/analysis"))
    parser.add_argument("--output", type=Path, default=Path("results/figures"))
    parser.add_argument(
        "--native-figures",
        action="store_true",
        help="Use native plot layout without the final manuscript typography",
    )
    args = parser.parse_args()
    configure()
    render(args.analysis_root, args.output)
    stem = Path(render.__code__.co_filename).stem
    finish_publication(args.output / (stem + ".svg"), native=args.native_figures)


def save_frontier_figure(
    figure,
    output_dir,
    name,
    rows,
    *,
    pad_inches=0.1,
    extra_canvas=(0, 0),
    panel_offsets=None,
):
    """Finish the native vector gradient and provider marks without moving data."""
    import pymupdf
    from .vector_style import pareto_embed_brandmarks

    save_figure(
        figure, output_dir, name, pad_inches=pad_inches, extra_canvas=extra_canvas
    )
    svg = Path(output_dir) / f"{name}.svg"
    pareto_embed_brandmarks(svg, rows)
    if panel_offsets:
        import xml.etree.ElementTree as ET
        import cairosvg

        root = ET.parse(svg).getroot()
        for identifier, (x, y) in panel_offsets.items():
            group = root.find(f".//{{http://www.w3.org/2000/svg}}g[@id='{identifier}']")
            if group is None:
                raise ValueError(f"Missing figure panel: {identifier}")
            group.set("transform", f"translate({x} {y})")
        svg.write_bytes(ET.tostring(root))
        cairosvg.svg2pdf(
            bytestring=svg.read_bytes(), write_to=str(svg.with_suffix(".pdf"))
        )
    with pymupdf.open(svg.with_suffix(".pdf")) as document:
        document[0].get_pixmap(
            matrix=pymupdf.Matrix(160 / 72, 160 / 72), alpha=False
        ).save(svg.with_suffix(".png"))


def finish_publication(svg_path, native=False):
    """Apply the paper's final typography and record unchanged scientific geometry.

    Native mode is useful for changed data or an alternative plotting layout.
    Missing Arial leaves the native figure intact and is recorded explicitly.
    Other styling errors fail visibly instead of silently dropping publication style.
    """
    import hashlib
    import xml.etree.ElementTree as ET
    from .publication_style import apply, scientific_geometry
    from .journal import write_json as write_receipt

    svg_path = Path(svg_path)
    profile = (
        Path(__file__).resolve().parents[1]
        / "figures/styles"
        / (svg_path.stem + ".json")
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    has_arial = "Arial" in available
    annotation_ids = []
    profile_hash = None
    if profile.exists():
        profile_hash = hashlib.sha256(profile.read_bytes()).hexdigest()
        specification = json.loads(profile.read_text())
        annotation_ids = [
            ident
            for pair in specification.get("straight_annotation_leaders", [])
            for ident in pair
        ]
        annotation_ids += [
            box["id"] for box in specification.get("label_backgrounds", [])
        ]
    before = scientific_geometry(ET.parse(svg_path).getroot(), annotation_ids)
    if native:
        result = {
            "status": "native_requested",
            "reason": "Publication typography disabled by --native-figures.",
        }
    elif not has_arial:
        result = {
            "status": "native_font_fallback",
            "reason": "Arial is unavailable; retained native font fallback output.",
        }
        print(
            f"  {svg_path.stem}: Arial unavailable; saved native font fallback.",
            flush=True,
        )
    elif svg_path.stem.startswith("figure_06_"):
        result = {
            "status": "native_publication_layout",
            "reason": "Figure 6 implements the final manuscript layout directly.",
        }
    else:
        result = {"status": "publication_style", **apply(svg_path, profile)}
    after = scientific_geometry(ET.parse(svg_path).getroot(), annotation_ids)
    if before != after:
        raise ValueError(f"{svg_path.stem}: final styling changed scientific geometry")
    result.update(
        figure=svg_path.stem,
        profile_sha256=profile_hash,
        arial_available=has_arial,
        scientific_geometry_before_sha256=before,
        scientific_geometry_after_sha256=after,
        scientific_geometry_unchanged=True,
        svg_sha256=hashlib.sha256(svg_path.read_bytes()).hexdigest(),
        pdf_sha256=hashlib.sha256(
            svg_path.with_suffix(".pdf").read_bytes()
        ).hexdigest(),
    )
    write_receipt(svg_path.with_suffix(".style.json"), result)
    return result
