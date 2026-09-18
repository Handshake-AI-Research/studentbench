"""Verify the published data snapshot without storing a large per-file manifest."""

from pathlib import Path
import hashlib
import json

from .data import sha256
from .journal import Journal, write_json


def verify_data(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    expected = json.loads(
        (
            Path(__file__).resolve().parents[1] / "verification/data_snapshot.json"
        ).read_text()
    )
    paths = []
    for name in expected["data_roots"]:
        path = data_dir / name
        paths.extend(path.rglob("*") if path.is_dir() else [path])
    paths = sorted(
        (p for p in paths if p.is_file()),
        key=lambda p: p.relative_to(data_dir).as_posix(),
    )
    journal = Journal(output_dir / "input_hashes.jsonl")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(data_dir).as_posix()
        stat = path.stat()
        signature = (
            f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}"
        )
        row = journal.get(relative, signature)
        if row is None:
            row = dict(sha256=sha256(path))
            journal.save(relative, signature, row)
        digest.update(f"{relative}\t{row['sha256']}\n".encode())
    result = dict(
        complete=True,
        file_count=len(paths),
        sha256=digest.hexdigest(),
        passed=len(paths) == expected["file_count"]
        and digest.hexdigest() == expected["sha256"],
    )
    write_json(output_dir / "data_snapshot.json", result)
    if not result["passed"]:
        raise ValueError(
            "Dataset differs from the paper release; see verification/data_snapshot.json"
        )
    return result
