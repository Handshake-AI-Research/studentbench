"""Small, resumable output helpers. Source data are never written."""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import numpy as np


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


class Journal:
    """Append each completed item, resuming only identical input/code fingerprints."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows = {}
        if self.path.exists():
            lines = self.path.read_bytes().splitlines(keepends=True)
            offset = 0
            for index, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if index != len(lines) - 1 or line.endswith(b"\n"):
                        raise
                    # A killed append can leave an unfinished final record. Keep
                    # it for inspection and resume from the last complete item.
                    self.path.with_suffix(self.path.suffix + ".incomplete").write_bytes(
                        line
                    )
                    with self.path.open("r+b") as handle:
                        handle.truncate(offset)
                    break
                self.rows[row["key"]] = row
                offset += len(line)
                if index == len(lines) - 1 and not line.endswith(b"\n"):
                    with self.path.open("ab") as handle:
                        handle.write(b"\n")

    def get(self, key, signature):
        row = self.rows.get(key)
        return row["value"] if row and row["signature"] == signature else None

    def save(self, key, signature, value):
        row = dict(key=key, signature=signature, value=plain(value))
        with self.path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows[key] = row
        return row["value"]
