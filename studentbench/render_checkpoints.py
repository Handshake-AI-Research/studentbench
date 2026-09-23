"""Resume completed figure/table exports only when inputs and outputs still match."""

from pathlib import Path
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from .data import sha256
from .journal import Journal


def render_environment():
    """Identify the actual renderer and fonts, including font fallback changes."""
    from matplotlib import font_manager

    packages = {}
    for name in ["matplotlib", "numpy", "scipy", "Pillow", "fonttools",
                 "PyMuPDF", "CairoSVG", "cairocffi", "pypdf", "adjustText"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        import cairocffi
        cairo = cairocffi.cairo_version_string()
    except (ImportError, OSError):
        cairo = None
    available = {font.name for font in font_manager.fontManager.ttflist}
    family = "Arial" if "Arial" in available else "DejaVu Sans"
    fonts = {}
    for style, weight in [("normal", "normal"), ("normal", "bold"),
                          ("italic", "normal"), ("italic", "bold")]:
        properties = font_manager.FontProperties(family=family, style=style, weight=weight)
        path = font_manager.findfont(properties, fallback_to_default=False)
        fonts[style + ":" + weight] = sha256(Path(path))
    return dict(python=platform.python_version(), platform=platform.system(),
                packages=packages, cairo=cairo, font_family=family, fonts=fonts)


class RenderCheckpoints:
    def __init__(self, root, output, native=False):
        self.root, self.output = Path(root), Path(output)
        self.hashes = Journal(self.output / 'render_input_hashes.jsonl')
        self.completed = Journal(self.output / 'render_checkpoints.jsonl')
        sources = [self.root / 'reproduce.py', self.root / 'pyproject.toml']
        for directory in ['studentbench', 'figures', 'tables', 'assets']:
            sources.extend(p for p in (self.root / directory).rglob('*')
                           if p.is_file() and '__pycache__' not in p.parts)
        environment = json.dumps(render_environment(), sort_keys=True).encode()
        self.source_signature = (
            self.digest(sources) + str(native) + hashlib.sha256(environment).hexdigest()
        )

    def digest(self, paths):
        result = hashlib.sha256()
        for path in sorted(map(Path, paths)):
            stat = path.stat()
            signature = f'{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}'
            key = str(path.resolve())
            value = self.hashes.get(key, signature)
            if value is None:
                value = self.hashes.save(key, signature, sha256(path))
            result.update(f'{path}\t{value}\n'.encode())
        return result.hexdigest()

    def run(self, key, inputs, outputs, build):
        outputs = list(map(Path, outputs))
        signature = self.source_signature + self.digest(inputs)
        previous = self.completed.get(key, signature)
        if previous and all(path.is_file() and sha256(path) == previous.get(path.name)
                            for path in outputs):
            print(f'  {key}: already complete', flush=True)
            return
        build()
        # Hashing all expected files also rejects incomplete exports.
        self.completed.save(key, signature, {p.name: sha256(p) for p in outputs})
