"""Preserve vector text and the paper's frontier badges and gradient backgrounds."""

from pathlib import Path
import copy
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import cairosvg
from .plotting import PARETO_BLUE

FIGURE_FONT = "Arial"
BRANDMARKS = Path(__file__).resolve().parents[1] / "assets/brandmarks"


def _svg_geometry(root):
    """Fingerprint all non-typographic attributes, geometry, and exact strings."""
    rows = []
    for e in root.iter():
        attrs = {
            k: v
            for k, v in e.attrib.items()
            if not k.startswith("font-")
            and k != "{http://www.w3.org/XML/1998/namespace}space"
        }
        if "style" in attrs:
            declarations = dict(
                (k.strip(), v.strip())
                for k, v in re.findall(r"([^:;]+):([^;]+)", attrs["style"])
            )
            attrs["style"] = json.dumps(
                sorted(
                    (k, v)
                    for k, v in declarations.items()
                    if not re.match(r"font(?:-[\w-]+)?$", k)
                )
            )
            if attrs["style"] == "[]":
                attrs.pop("style")
        rows.append((e.tag, sorted(attrs.items()), e.text, e.tail))
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


def normalize_svg_typography(raw):
    """Change font declarations only, retaining SVG text and mark coordinates."""
    root = ET.fromstring(raw)
    before = _svg_geometry(root)
    ns = "{http://www.w3.org/2000/svg}"
    for e in root.iter():
        style = dict(
            (k.strip(), v.strip())
            for k, v in re.findall(r"([^:;]+):([^;]+)", e.get("style", ""))
        )
        if "font" in style:
            match = re.fullmatch(
                r"(.*?)\s*(\d+(?:\.\d+)?(?:px|pt|em|%))(?:/\S+)?\s+(.+)",
                style.pop("font"),
            )
            assert match is not None, "Unrecognized SVG font shorthand"
            prefix, size, _ = match.groups()
            style.setdefault("font-size", size)
            for token in prefix.split():
                if token in ["italic", "oblique"]:
                    style.setdefault("font-style", token)
                elif token in ["bold", "bolder", "lighter"] or token.isdigit():
                    style.setdefault("font-weight", token)
        text = e.tag in [ns + "text", ns + "tspan"]
        if text or "font-family" in e.attrib or "font-family" in style:
            e.set("font-family", FIGURE_FONT)
            style["font-family"] = FIGURE_FONT
        for key in ["font-size", "font-weight", "font-style"]:
            if key in style:
                e.set(key, style[key])
        if text:
            e.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        if style:
            e.set("style", "; ".join(k + ": " + v for k, v in style.items()))
    assert _svg_geometry(root) == before, (
        "Typography normalization altered geometry or text"
    )
    return ET.tostring(root, encoding="utf-8"), before


def pareto_embed_brandmarks(svg_path, metadata):
    ns = {"s": "http://www.w3.org/2000/svg"}
    root = ET.fromstring(normalize_svg_typography(svg_path.read_bytes())[0])
    # Native SVG gradients become PDF vector shadings, keeping all text and
    # marks searchable and sharp without introducing a raster background.
    defs = root.find("s:defs", ns)
    if defs is None:
        defs = ET.SubElement(root, "{" + ns["s"] + "}defs")
    for wash in root.findall(".//s:g", ns):
        if not wash.get("id", "").startswith("pareto_wash_"):
            continue
        ident = wash.get("id") + "_fade"
        # Engagement is maximized, so its wash fades away from the right-hand
        # frontier; cost and latency are minimized and fade toward the right.
        reverse = wash.get("id").endswith("_engagement")
        gradient = ET.SubElement(
            defs,
            "{" + ns["s"] + "}linearGradient",
            {
                "id": ident,
                "x1": "100%" if reverse else "0%",
                "y1": "0%",
                "x2": "0%" if reverse else "100%",
                "y2": "0%",
            },
        )
        for offset, opacity in [
            ("0%", ".075"),
            ("35%", ".045"),
            ("70%", ".012"),
            ("100%", "0"),
        ]:
            ET.SubElement(
                gradient,
                "{" + ns["s"] + "}stop",
                {"offset": offset, "stop-color": PARETO_BLUE, "stop-opacity": opacity},
            )
        for e in wash.iter():
            if "style" in e.attrib and "fill:" in e.get("style", ""):
                e.set("style", f"fill: url(#{ident}); stroke: none")
    files = {
        "Google": "google.svg",
        "OpenAI": "openai.svg",
        "Anthropic": "anthropic.svg",
        "Moonshot": "moonshot.svg",
    }
    for row in metadata:
        if not row["frontier"]:
            continue
        holder = root.find(f".//s:g[@id='{row['icon_gid']}']", ns)
        assert holder is not None
        path = holder.find("s:path", ns)
        assert path is not None
        points = list(
            map(
                float,
                re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", path.get("d")),
            )
        )
        xs = points[::2]
        ys = points[1::2]
        x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
        source = BRANDMARKS / files[row["family"]]
        logo = ET.parse(source).getroot()
        vx, vy, vw, vh = map(float, logo.get("viewBox").split())
        children = list(logo)
        if row["family"] == "OpenAI":
            # The official avatar surrounds the Blossom with white padding.
            # Fit its unchanged black path using the official inner mask bounds.
            mask = min(
                logo.findall(".//s:mask", ns),
                key=lambda e: float(e.get("width")) * float(e.get("height")),
            )
            vx, vy, vw, vh = [float(mask.get(k)) for k in ["x", "y", "width", "height"]]
            children = [logo.find(".//s:path[@fill='black']", ns)]
            assert children[0] is not None
        scale = min(w / vw, h / vh)
        group = ET.Element(
            "{" + ns["s"] + "}g",
            {
                "transform": f"translate({x + (w - vw * scale) / 2} {y + (h - vh * scale) / 2}) scale({scale}) translate({-vx} {-vy})",
                "color": logo.get("color", "#141413"),
                "fill": logo.get("fill", "black"),
            },
        )
        for child in children:
            group.append(copy.deepcopy(child))
        label = root.find(f".//s:g[@id='{row['label_gid']}']/s:text", ns)
        assert label is not None
        label.text = row["name"]
        label.set("x", str(x + w + 3.2))
        label.attrib.pop("transform", None)
        label.set(
            "style",
            re.sub(
                r"text-anchor:\s*[^;]+", "text-anchor: start", label.get("style", "")
            ),
        )
        ids = {
            node.get("id"): row["icon_gid"] + "_" + node.get("id")
            for node in group.iter()
            if node.get("id")
        }
        for node in group.iter():
            if node.get("id"):
                node.set("id", ids[node.get("id")])
            for attr, value in list(node.attrib.items()):
                for old, new in ids.items():
                    value = value.replace("url(#" + old + ")", "url(#" + new + ")")
                    if attr.endswith("href") and value == "#" + old:
                        value = "#" + new
                node.set(attr, value)
        holder.remove(path)
        holder.append(group)
        row["brandmark"] = source.name
        row["brandmark_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    raw = ET.tostring(root, encoding="unicode")
    svg_path.write_text(raw)
    cairosvg.svg2pdf(
        bytestring=raw.encode(), write_to=str(svg_path.with_suffix(".pdf"))
    )
