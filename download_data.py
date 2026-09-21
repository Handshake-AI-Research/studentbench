#!/usr/bin/env python3
"""Download the public study records; Hugging Face resumes interrupted transfers."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument(
        "--revision",
        help="Dataset revision (defaults to the version used for the paper)",
    )
    args = parser.parse_args()
    pin = json.loads(
        (Path(__file__).parent / "verification/data_snapshot.json").read_text()
    )
    revision = args.revision or pin.get("huggingface_revision")
    if not revision:
        parser.error(
            "No dataset revision is configured. Supply --revision with a Hugging Face "
            "commit hash, or point reproduce.py --data to an existing download."
        )
    path = snapshot_download(
        repo_id=pin["dataset"],
        repo_type="dataset",
        revision=revision,
        local_dir=args.output,
    )
    print(f"Downloaded {pin['dataset']} at {revision}: {path}")


if __name__ == "__main__":
    main()
