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
            page.show_pdf_page(rendered[0].rect, rendered, 0)
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
