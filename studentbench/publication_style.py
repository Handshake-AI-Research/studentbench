"""Apply the paper's typography to newly calculated vector figures.

Each figure has an inspectable layout file in figures/styles. These files store
font declarations, label coordinates, canvas sizes and logo positions. They
contain neither measured values nor scientific paths. Label characters always
come from the new plot; a changed value fails the label check instead of being
silently replaced. Data marks are fingerprinted before and after styling.
"""

from pathlib import Path
import copy
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET

import cairosvg
import pymupdf

from .data import sha256

SVG = "{http://www.w3.org/2000/svg}"
ROOT = Path(__file__).resolve().parents[1]
NUMBERS = r"[-+]?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def visible_text(element):
    return "".join("".join(text.itertext()) for text in element.iter(SVG + "text"))


def scientific_geometry(root, annotation_ids=()):
    """Include every data shape and its inherited transforms and clipping."""
    shapes = {
        SVG + tag
        for tag in [
            "path",
            "line",
            "use",
            "rect",
            "circle",
            "ellipse",
            "polyline",
            "polygon",
            "image",
        ]
    }
    records = []

    def visit(element, inherited=()):
        ident = element.get("id", "")
        if ident in {
            "patch_1",
            "studentbench_provider_marks",
            "author_font_choice_inset_leaders",
            *annotation_ids,
        } or ident.startswith(("legend_", "label_guide_", "pareto_icon_")):
            return
        inherited = inherited + tuple(
            (key, element.get(key))
            for key in ["transform", "clip-path"]
            if element.get(key)
        )
        if element.tag in shapes:
            records.append((element.tag, element.attrib, inherited))
        for child in element:
            visit(child, inherited)

    visit(root)
    return digest(json.dumps(records, sort_keys=True).encode())


def styled_text(template, characters, original):
    """Fill formatting-only character slots using this plot's actual text."""
    cursor = 0
    originals = {
        element.get("id"): element for element in original.iter() if element.get("id")
    }

    def field(specification):
        nonlocal cursor
        if "whitespace" in specification:
            return specification["whitespace"]
        count = specification["characters"]
        text = characters[cursor : cursor + count]
        cursor += count
        return text or None

    def build(specification):
        if "preserve_element" in specification:
            return copy.deepcopy(originals[specification["preserve_element"]])
        if "preserve_element_tag" in specification:
            raise ValueError("Unidentified annotation background")
        node = ET.Element(SVG + specification["tag"], specification["attributes"])
        node.text = field(specification["text"])
        for child in specification["children"]:
            node.append(build(child))
        node.tail = field(specification["tail"])
        return node

    result = build(template)
    if cursor != len(characters) or visible_text(result) != characters:
        raise ValueError("Typography template would change generated label text")
    return result


def provider_marks(specifications):
    """Draw the same four provider logos used by the native frontier figures."""
    container = ET.Element(SVG + "g", {"id": "studentbench_provider_marks"})
    for specification in specifications:
        brand = specification["brand"]
        if brand not in {"google", "openai", "anthropic", "moonshot"}:
            raise ValueError(f"Unknown provider artwork: {brand}")
        logo = ET.parse(ROOT / "assets/brandmarks" / (brand + ".svg")).getroot()
        children = list(logo)
        if brand == "openai":
            # The official avatar contains padding. Use its unchanged black mark.
            children = [logo.find(".//" + SVG + "path[@fill='black']")]
        group = ET.SubElement(container, SVG + "g", specification["attributes"])
        for child in children:
            if child is None:
                raise ValueError(f"Missing provider artwork: {brand}")
            group.append(copy.deepcopy(child))
    return container


def straighten_annotation_leaders(root, identifiers, mappings):
    """Connect existing labels to their freshly plotted points, with a 4pt gap."""
    if not mappings or mappings[0][0] not in identifiers:
        return
    parents = {child: parent for parent in root.iter() for child in parent}
    container = parents[identifiers[mappings[0][0]]]
    slot = list(container).index(identifiers[mappings[0][0]])
    markers = [
        (float(z.get("x")), float(z.get("y")))
        for z in root.iter(SVG + "use")
        if z.get("x") and z.get("y")
    ]
    groups = []
    for source, destination in mappings:
        group = identifiers[source]
        line = group.find(SVG + "path")
        coordinates = list(map(float, re.findall(NUMBERS, line.get("d"))))
        label, near = coordinates[:2], coordinates[-2:]
        point = min(markers, key=lambda xy: math.dist(xy, near))
        if abs(math.dist(point, near) - 4) >= 0.01:
            raise ValueError("Annotation endpoint no longer matches a plotted point")
        distance = math.dist(point, label)
        start = [point[i] + 4 * (label[i] - point[i]) / distance for i in [0, 1]]
        line.set(
            "d", f"M {start[0]:.6f} {start[1]:.6f} L {label[0]:.6f} {label[1]:.6f}"
        )
        group.set("id", destination)
        groups.append(group)
        container.remove(group)
    for index, group in enumerate(groups):
        container.insert(slot + index, group)


def apply(svg_path: Path, profile_path: Path | None = None):
    """Style one completed plot, asserting unchanged scientific marks and text."""
    svg_path = Path(svg_path)
    profile_path = profile_path or ROOT / "figures/styles" / (svg_path.stem + ".json")
    profile = json.loads(Path(profile_path).read_text())
    root = ET.parse(svg_path).getroot()
    identifiers = {z.get("id"): z for z in root.iter() if z.get("id")}
    mappings = profile.get("straight_annotation_leaders", [])
    annotations = [ident for pair in mappings for ident in pair] + [
        box["id"] for box in profile.get("label_backgrounds", [])
    ]
    geometry_before = scientific_geometry(root, annotations)
    text_before = visible_text(root)
    root.attrib.update(profile["canvas"])
    used = set()
    for item in profile["text_groups"]:
        target = identifiers.get(item["id"])
        if (
            target is None
            or digest(visible_text(target).encode()) != item["text_sha256"]
        ):
            matches = [
                node
                for node in root.iter(SVG + "g")
                if node.get("id", "").startswith("text_")
                and node not in used
                and digest(visible_text(node).encode()) == item["text_sha256"]
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{svg_path.stem}: generated label differs at {item['id']}"
                )
            target = matches[0]
        used.add(target)
        replacement = styled_text(item["template"], visible_text(target), target)
        target.attrib.clear()
        target.attrib.update(replacement.attrib)
        target.text, target.tail = replacement.text, replacement.tail
        target[:] = list(replacement)
    for legend in profile["legends"]:
        group = identifiers[legend["id"]]
        if legend["transform"] is None:
            group.attrib.pop("transform", None)
        else:
            group.set("transform", legend["transform"])
        markers = list(group.iter(SVG + "use"))
        if len(markers) != len(legend["marker_positions"]):
            raise ValueError("Legend marker count differs from finalized layout")
        for marker, position in zip(markers, legend["marker_positions"]):
            marker.attrib.update(position)
        if profile.get("align_legend_lines"):
            for line_group in group.iter(SVG + "g"):
                line = line_group.find(SVG + "path")
                markers = list(line_group.iter(SVG + "use"))
                if (
                    line is None
                    or len(markers) != 1
                    or not line_group.get("id", "").startswith("line2d_")
                ):
                    continue
                values = list(map(float, re.findall(NUMBERS, line.get("d"))))
                if len(values) != 6 or values[1] != values[3] or values[3] != values[5]:
                    raise ValueError("Unexpected legend stroke shape")
                y = markers[0].get("y")
                offset = float(markers[0].get("x")) - values[2]
                values = [
                    value + offset if i % 2 == 0 else value
                    for i, value in enumerate(values)
                ]
                line.set(
                    "d",
                    f"M {values[0]:.6f} {y} L {values[2]:.6f} {y} L {values[4]:.6f} {y}",
                )
    if "studentbench_provider_marks" in identifiers:
        for parent in root.iter():
            if identifiers["studentbench_provider_marks"] in list(parent):
                parent.remove(identifiers["studentbench_provider_marks"])
                break
    if profile["provider_marks"]:
        root.append(provider_marks(profile["provider_marks"]))
    if svg_path.stem == "figure_01_learning":
        inset = root.find(
            ".//" + SVG + "g[@id='adjusted_cluster_aware_human_comparison']"
        )
        root.remove(inset)
        root.append(inset)
    straighten_annotation_leaders(root, identifiers, mappings)
    # White annotation backings and logos are typography, never data marks.
    for box in profile.get("label_backgrounds", []):
        group = root.find(".//" + SVG + "g[@id='" + box["id"] + "']")
        path = group.find(SVG + "path")
        style = path.get("style", "")
        if not (
            "fill: #ffffff; opacity: 0.87" in style
            or "fill: #ffffff; opacity: 0.97; stroke: #e4e9ed;" in style
        ):
            raise ValueError("Unexpected label background style")
        left, top, right, bottom = box["bounds"]
        radius = box["corner_radius"]
        if radius:
            start, end = (top, bottom) if box["start_at_top"] else (bottom, top)
            direction = 1 if box["start_at_top"] else -1
            path.set(
                "d",
                f"M {left + radius} {start} L {right - radius} {start} Q {right} {start} {right} {start + direction * radius} L {right} {end - direction * radius} Q {right} {end} {right - radius} {end} L {left + radius} {end} Q {left} {end} {left} {end - direction * radius} L {left} {start + direction * radius} Q {left} {start} {left + radius} {start} z",
            )
        else:
            path.set(
                "d",
                f"M {left} {bottom} L {right} {bottom} L {right} {top} L {left} {top} z",
            )
    for ident, transforms in profile.get("frontier_badge_transforms", {}).items():
        groups = [child for child in identifiers[ident] if child.tag == SVG + "g"]
        if len(groups) != len(transforms):
            raise ValueError("Frontier badge structure differs")
        for group, transform in zip(groups, transforms):
            if transform is None:
                group.attrib.pop("transform", None)
            else:
                group.set("transform", transform)
    if profile.get("inset_label_guides"):
        parent = identifiers[profile["inset_label_guides"][0]["parent"]]
        for child in list(parent):
            if child.get("id") == "author_font_choice_inset_leaders":
                parent.remove(child)
        guides = ET.SubElement(
            identifiers[profile["inset_label_guides"][0]["parent"]],
            SVG + "g",
            {"id": "author_font_choice_inset_leaders"},
        )
        for guide in profile["inset_label_guides"]:
            ET.SubElement(
                guides,
                SVG + "polyline",
                {
                    "points": " ".join(f"{x},{y}" for x, y in guide["points"]),
                    "fill": "none",
                    "stroke": guide["stroke"],
                    "stroke-width": str(guide["stroke_width"]),
                },
            )
    width, height = profile["background_size"]
    background = identifiers["patch_1"].find(SVG + "path")
    if background.get("style") != "fill: #ffffff":
        raise ValueError("Unexpected page background")
    background.set("d", f"M 0 {height}  L {width} {height}  L {width} 0  L 0 0  z ")
    if scientific_geometry(root, annotations) != geometry_before:
        raise ValueError("Publication styling changed scientific geometry")
    if visible_text(root) != text_before:
        raise ValueError("Publication styling changed generated visible text")
    raw = ET.tostring(root)
    # The original figures used a fixed blank page containing vector Cairo output.
    # The new PDF has no run-time dates or random document ID.
    with pymupdf.open(
        stream=cairosvg.svg2pdf(bytestring=raw), filetype="pdf"
    ) as rendered:
        with pymupdf.open() as document:
            _, _, width, height = profile["pdf_canvas"]
            page = document.new_page(width=width, height=height)
            page.show_pdf_page(
                page.rect if profile.get("fit_pdf_canvas") else rendered[0].rect,
                rendered,
                0,
            )
            temporary = svg_path.with_suffix(".styled.tmp.pdf")
            document.save(temporary, garbage=4, deflate=True, no_new_id=True)
    svg_path.write_bytes(raw)
    temporary.replace(svg_path.with_suffix(".pdf"))
    with pymupdf.open(svg_path.with_suffix(".pdf")) as document:
        document[0].get_pixmap(
            matrix=pymupdf.Matrix(160 / 72, 160 / 72), alpha=False
        ).save(svg_path.with_suffix(".png"))
    return dict(
        figure=profile["figure"],
        profile_sha256=sha256(Path(profile_path)),
        scientific_geometry_sha256=geometry_before,
        scientific_geometry_unchanged=True,
        generated_text_unchanged=True,
        svg_sha256=sha256(svg_path),
        pdf_sha256=sha256(svg_path.with_suffix(".pdf")),
    )


def _pdf_font(bold=False):
    from matplotlib.font_manager import FontProperties, findfont

    return findfont(
        FontProperties(family="Arial", weight="bold" if bold else "normal"),
        fallback_to_default=False,
    )


def color(c):
    return ((c >> 16 & 255) / 255, (c >> 8 & 255) / 255, (c & 255) / 255)


def lines(page):
    return [l for b in page.get_text("dict")["blocks"] for l in b.get("lines", [])]


def spans(page):
    return [s for l in lines(page) for s in l["spans"]]


def replace(page, old, new, align="left"):
    found = [s for s in spans(page) if s["text"] == old]
    assert found, (old, [s["text"] for s in spans(page)])
    todo = []
    for s in found:
        r = pymupdf.Rect(s["bbox"])
        font = _pdf_font(True) if "Bold" in s["font"] else _pdf_font(False)
        name = "replacementBold" if "Bold" in s["font"] else "replacementArial"
        x, y = s["origin"]
        fw = pymupdf.Font(fontfile=font).text_length(new, fontsize=s["size"])
        if align == "right":
            x = r.x1 - fw
        elif align == "center":
            x = (r.x0 + r.x1 - fw) / 2
        page.add_redact_annot(r, fill=False, cross_out=False)
        todo.append((s, new, (x, y), font, name))
    page.apply_redactions(images=0, graphics=0, text=0)
    for s, t, origin, font, name in todo:
        page.insert_text(
            origin,
            t,
            fontname=name,
            fontfile=font,
            fontsize=s["size"],
            color=color(s["color"]),
        )
    return len(todo)


def remove_text(page, terms):
    count = 0
    for s in spans(page):
        if any(t in s["text"] for t in terms):
            page.add_redact_annot(s["bbox"], fill=False, cross_out=False)
            count += 1
    page.apply_redactions(images=0, graphics=0, text=0)
    return count


def primary(doc):
    p = doc[0]
    assert ".023" not in p.get_text(), "Unexpected second equivalence p label"
    replace(p, "Highest observed AI mean per topic", "Highest AI mean per topic")
    replace(
        p, "Adjusted AI −human (percentage points)", "AI − human (percentage points)"
    )
    replace(p, "90% cluster-aware CI", "90% CI", align="center")
    return doc, {
        "edits": [
            "Remove observed/adjusted/cluster-aware qualifiers from three labels. Existing inset has no .023 p label."
        ],
        "retained_data": "All plotted geometry, numeric labels, intervals, and colors unchanged.",
    }


def quartile(doc):
    # Remove painting operations in the raw-series color in nested PDF forms.
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import ContentStream
    import io

    reader = PdfReader(io.BytesIO(doc.tobytes()))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    raw = (134 / 255, 147 / 255, 157 / 255)
    seen = set()
    removed = 0

    def israw(v):
        return len(v) == 3 and max(abs(a - b) for a, b in zip(v, raw)) < 2e-5

    def walk(resources):
        nonlocal removed
        for ref in resources.get("/XObject", {}).values():
            ob = ref.get_object()
            if ref.idnum in seen:
                continue
            seen.add(ref.idnum)
            if ob.get("/Subtype") != "/Form":
                continue
            stream = ContentStream(ob, writer)
            fill = stroke = (0.0,)
            stack = []
            changed = False
            new = []
            for args, op in stream.operations:
                if op == b"q":
                    stack.append((fill, stroke))
                elif op == b"Q":
                    fill, stroke = stack.pop()
                elif op in (b"rg", b"g", b"k"):
                    fill = tuple(float(v) for v in args)
                elif op in (b"RG", b"G", b"K"):
                    stroke = tuple(float(v) for v in args)
                if (
                    op in (b"S", b"s")
                    and israw(stroke)
                    or op in (b"f", b"F", b"f*")
                    and israw(fill)
                    or op in (b"B", b"B*", b"b", b"b*")
                    and (israw(fill) or israw(stroke))
                ):
                    op = b"n"
                    args = []
                    removed += 1
                    changed = True
                new.append((args, op))
            if changed:
                stream.operations = new
                ob.set_data(stream.get_data())
            walk(ob.get("/Resources", {}))

    walk(writer.pages[0]["/Resources"])
    assert (
        removed == 36
    ), removed  # 8 data circles, 16 caps, 8 CI segments, plus 4 legend glyphs.
    buf = io.BytesIO()
    writer.write(buf)
    out = pymupdf.open(stream=buf.getvalue(), filetype="pdf")
    p = out[0]
    assert remove_text(p, ["Observed", "Baseline-adjusted"]) == 2
    # Remove only the now-unnecessary blue legend key; retain every data path.
    # This location contains the legend glyph, not a data interval.
    p.add_redact_annot(pymupdf.Rect(715, 214, 732, 224), fill=False, cross_out=False)
    p.apply_redactions(images=0, graphics=1, text=0)
    return out, {
        "edits": [
            "Remove gray raw-Welch series and redundant two-estimator legend. Retain blue ANCOVA-HC3 series."
        ],
        "raw_paint_operations_removed": removed,
        "retained_data": "All blue data coordinates, CIs, axes, scales, colors, and positions unchanged.",
    }


def dialogue(doc):
    count = 0
    for old, new in [("Quant: Holm ", "Quant: "), ("Verbal: Holm ", "Verbal: ")]:
        count += replace(doc[0], old, new, align="right")
    assert count == 6
    return doc, {
        "edits": [
            "Remove Holm from six p labels; ordinary p notation retains identical numeric values."
        ],
        "retained_data": "Every point, curve, band, p value, and annotation endpoint preserved.",
    }


def cost(doc):
    replace(
        doc[0],
        "Highlighted rows: human-equivalent gains",
        "Passed individual equivalence tests",
    )
    return doc, {
        "edits": ["Legend reads Passed individual equivalence tests."],
        "retained_data": "All four highlights, colors, bars, values, inset, and table entries unchanged.",
    }


def latency(doc):
    p = doc[0]
    found = [
        l for l in lines(p) if "Pearson r=" in "".join(s["text"] for s in l["spans"])
    ]
    assert len(found) == 3, len(found)
    for l in found:
        p.add_redact_annot(l["bbox"], fill=False, cross_out=False)
    p.apply_redactions(images=0, graphics=0, text=0)
    return doc, {
        "edits": [
            "Remove three Pearson annotations; retain existing Spearman rho and p annotations."
        ],
        "retained_data": "Spearman values, scatter positions, model labels, and fitted lines unchanged.",
    }


def activity(doc):
    old = "Combined sections · rows ordered by student-message volume · cells show observed values"
    replace(doc[0], old, old.replace("observed ", ""))
    return doc, {
        "edits": ["Remove observed from cell-value description."],
        "retained_data": "All heatmap cells, values, ordering, colors, and borders unchanged.",
    }


def domains(doc):
    replace(doc[0], "Observed learning gain", "Learning gain")
    return doc, {
        "edits": ["Remove Observed from learning-gain legend."],
        "retained_data": "All seven domain plots, numerical values, geometry, colors, ordering, and provider marks unchanged.",
    }


def finish_paper_pdf(svg_path):
    """Apply the paper's final presentation edits to newly generated vectors."""
    svg_path = Path(svg_path)
    operations = {
        "figure_01_learning": primary,
        "figure_05_cost_per_gain": cost,
        "figure_06_engagement_practice": dialogue,
        "figure_13_domain_learning": domains,
        "figure_14_starting_proficiency": quartile,
        "figure_18_reply_time_engagement": latency,
        "figure_19_tutoring_experience": activity,
    }
    operation = operations.get(svg_path.stem)
    if operation is None:
        return
    pdf = svg_path.with_suffix(".pdf")
    with pymupdf.open(pdf) as document:
        original_marks = document[0].get_drawings()
        revised, presentation = operation(document)
        revised_marks = revised[0].get_drawings()
        if operation is quartile:

            def retained_blue(marks):
                blue = (8 / 255, 124 / 255, 167 / 255)
                legend = pymupdf.Rect(715, 214, 732, 224)
                return [
                    {key: value for key, value in mark.items() if key != "seqno"}
                    for mark in marks
                    if not legend.contains(mark["rect"])
                    and any(
                        color is not None
                        and len(color) == 3
                        and max(abs(a - b) for a, b in zip(color, blue)) < 1e-4
                        for color in (mark["fill"], mark["color"])
                    )
                ]

            assert retained_blue(original_marks) == retained_blue(revised_marks)
            presentation["retained_blue_marks"] = len(retained_blue(revised_marks))
        else:
            assert [
                {key: value for key, value in mark.items() if key != "seqno"}
                for mark in original_marks
            ] == [
                {key: value for key, value in mark.items() if key != "seqno"}
                for mark in revised_marks
            ], "Presentation changed plotted geometry"
        presentation["retained_marks_verified"] = True
        # Pin only serialization metadata; all marks and labels above were rebuilt.
        profile_path = ROOT / "figures/styles" / (svg_path.stem + ".json")
        profile = json.loads(profile_path.read_text()) if profile_path.exists() else {}
        if profile.get("pdf_document_id"):
            revised.xref_set_key(-1, "ID", profile["pdf_document_id"])
        temporary = pdf.with_suffix(".final.tmp.pdf")
        revised.save(temporary, garbage=4, deflate=True, no_new_id=True)
        # Match the vector SVG representation delivered with these edited PDFs.
        svg_path.write_text(revised[0].get_svg_image(text_as_path=False))
        revised[0].get_pixmap(
            matrix=pymupdf.Matrix(160 / 72, 160 / 72), alpha=False
        ).save(pdf.with_suffix(".png"))
        if revised is not document:
            revised.close()
    if svg_path.stem == "figure_05_cost_per_gain":
        _order_font_dictionary_keys(temporary)
    if profile.get("pdf_version"):
        raw = temporary.read_bytes()
        temporary.write_bytes(b"%PDF-" + profile["pdf_version"].encode() + raw[8:])
    temporary.replace(pdf)
    return presentation


def _order_font_dictionary_keys(pdf):
    """Keep the paper's PDF serialization without recompressing font streams."""
    with pymupdf.open(pdf) as document:
        streams = []
        for font in document[0].get_fonts(full=True):
            if "+" not in font[3]:
                continue
            references = [document.xref_get_key(font[0], "ToUnicode")]
            kind, reference = document.xref_get_key(font[0], "FontDescriptor")
            if kind == "xref":
                references.append(
                    document.xref_get_key(int(reference.split()[0]), "FontFile2")
                )
            streams.extend(
                int(value.split()[0]) for kind, value in references if kind == "xref"
            )
    raw = pdf.read_bytes()
    for xref in streams:
        pattern = rb"(?m)^" + str(xref).encode() + rb" 0 obj\n<<([^>]+)>>"
        match = re.search(pattern, raw)
        if not match:
            raise ValueError("Missing serialized font dictionary")
        body = match[1]
        fields = re.findall(rb"/Filter/FlateDecode|/Length1? [0-9]+", body)
        if b"".join(fields) != body:
            raise ValueError("Unexpected font stream dictionary")
        order = {b"/Filter": 0, b"/Length1": 1, b"/Length": 2}
        ordered = b"".join(
            sorted(
                fields,
                key=lambda field: order[
                    field.split(b" ")[0].removesuffix(b"/FlateDecode")
                ],
            )
        )
        assert len(body) == len(ordered)
        raw = raw[: match.start(1)] + ordered + raw[match.end(1) :]
    pdf.write_bytes(raw)
