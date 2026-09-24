#!/usr/bin/env python3
"""Reproduce StudentBench from the public dataset, one named stage at a time."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

from studentbench.journal import write_json

ROOT = Path(__file__).resolve().parent
ANALYSES = (
    "learning",
    "primary_equivalence",
    "costs",
    "teaching",
    "engagement",
    "engagement_robustness",
    "sensitivity",
    "reviewer",
    "tutor_dependence",
    "prompts",
    "geography",
    "data_quality",
    "leaderboards",
    "condition_interaction",
    "repeat_main",
    "study_summary",
)


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Downloaded StudentBench dataset directory",
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument(
        "--only",
        nargs="+",
        choices=(*ANALYSES, "figures", "tables", "verify"),
        help="Run selected stages; figures/tables require their analysis outputs",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare every published validation target",
    )
    parser.add_argument(
        "--verify-figure-bytes",
        action="store_true",
        help="Also require byte-identical figure files (needs the paper's fonts and rendering environment)",
    )
    parser.add_argument(
        "--native-figures",
        action="store_true",
        help="Use native plot layout without the final manuscript typography",
    )
    args = parser.parse_args()
    data, output = args.data.resolve(), args.output.resolve()
    if not (data / "studentbench_overall_parameters.json").is_file():
        parser.error(
            "--data must be the dataset root containing studentbench_overall_parameters.json"
        )
    if output == data or output.is_relative_to(data):
        parser.error("Output must lie outside the source dataset")
    output.mkdir(parents=True, exist_ok=True)
    analysis = output / "analysis"
    selected = args.only or [*ANALYSES, "figures", "tables"]
    if (args.verify or args.verify_figure_bytes) and "verify" not in selected:
        selected.append("verify")
    status = dict(
        complete=False,
        started_utc=datetime.now(timezone.utc).isoformat(),
        data=str(data),
        output=str(output),
        requested_stages=selected,
        stages=[],
    )
    write_json(output / "run.json", status)

    def record(event):
        with (output / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    for stage in selected:
        start = time.monotonic()
        print(f"[{stage}] starting", flush=True)
        record(dict(stage=stage, event="start", time=time.time()))
        try:
            if stage in ANALYSES:
                module = importlib.import_module(f"studentbench.{stage}")
                if stage in {"sensitivity", "engagement_robustness"}:
                    module.run(
                        data, analysis / stage, engagement_dir=analysis / "engagement"
                    )
                elif stage in {"repeat_main", "study_summary"}:
                    module.run(
                        data,
                        analysis / stage,
                        resource_dir=analysis / "costs",
                        conversation_dir=analysis / "teaching",
                    )
                else:
                    module.run(data, analysis / stage)
            elif stage == "figures":
                from studentbench.plotting import configure, finish_publication

                from studentbench.render_checkpoints import RenderCheckpoints

                configure()
                cache = RenderCheckpoints(ROOT, output, args.native_figures)
                inventory = json.loads((ROOT / "figures/manifest.json").read_text())
                for item in inventory["figures"]:
                    destinations = [output / "figures" / name for name in item["outputs"]]
                    if item["kind"] == "fixed_artwork":
                        destinations[0].parent.mkdir(parents=True, exist_ok=True)
                        cache.run(item["label"], [ROOT / item["static_asset"]], destinations,
                                  lambda: shutil.copy2(ROOT / item["static_asset"], destinations[0]))
                    else:
                        path = ROOT / item["script"]
                        print(f"  {path.stem}", flush=True)
                        def render():
                            load_module(path).render(analysis, output / "figures")
                            finish_publication(output / "figures" / (path.stem + ".svg"),
                                               native=args.native_figures)
                        cache.run(path.stem, [analysis / name for name in item["analysis_inputs"]],
                                  destinations, render)
                    record(dict(stage=stage, event="item_complete", item=item["label"]))
            elif stage == "tables":
                from studentbench.render_checkpoints import RenderCheckpoints

                cache = RenderCheckpoints(ROOT, output)
                inputs = [p for p in analysis.glob("*/*")
                          if p.is_file() and p.suffix in {".json", ".csv", ".jsonl"}]
                inputs.extend(p for p in data.glob("*.json") if p.is_file())
                # Tables A.1-A.3 also read profiles and assessment responses directly.
                inputs.extend(p for p in data.rglob("*") if p.is_file() and p.name in
                              {"student.json", "pretest_responses.csv", "posttest_responses.csv"})
                for path in sorted((ROOT / "tables").glob("table_*.py")):
                    print(f"  {path.stem}", flush=True)
                    destinations = [output / "tables" / (path.stem + suffix)
                                    for suffix in (".csv", ".md", ".tex", "_paper.tex")]
                    cache.run(path.stem, inputs, destinations,
                              lambda: load_module(path).run(data, analysis, output / "tables"))
                    record(dict(stage=stage, event="item_complete", item=path.stem))
            elif stage == "verify":
                from studentbench.snapshot import verify_data
                from studentbench.verification import verify_prompts_and_subgroups
                from studentbench.core_verification import (
                    verify_core, verify_primary_equivalence, verify_individual_equivalence,
                )
                from studentbench.verify_robustness import run as verify_robustness
                from studentbench.verify_tables import run as verify_tables
                from studentbench.condition_interaction import (
                    verify as verify_interaction,
                )
                from studentbench.figure_verification import (
                    verify_figures,
                    write_figure_manifest,
                    verify_rendered_figures,
                )
                from studentbench.verify_repeat_main import run as verify_repeat
                from studentbench.verify_study_summary import run as verify_study
                from studentbench.paper_results import run as verify_paper_results

                verify_data(data, output / "verification")
                verify_prompts_and_subgroups(analysis, output / "verification")
                verify_core(analysis, output / "verification")
                verify_primary_equivalence(analysis, output / "verification")
                verify_individual_equivalence(analysis, output / "verification")
                verify_robustness(analysis, output / "verification/robustness")
                verify_tables(output / "tables", output / "verification/tables")
                verify_interaction(analysis, output / "verification")
                verify_figures(analysis, output / "verification/figures")
                write_figure_manifest(analysis, output / "figures")
                if args.verify_figure_bytes:
                    verify_rendered_figures(output / "figures", output / "verification/figures")
                verify_repeat(analysis, output / "verification/repeat_main")
                verify_study(analysis, output / "verification/study_summary")
                verify_paper_results(analysis, output / "verification/paper_results", data)
            elapsed = round(time.monotonic() - start, 2)
            status["stages"].append(dict(stage=stage, complete=True, seconds=elapsed))
            record(dict(stage=stage, event="complete", seconds=elapsed))
            write_json(output / "run.json", status)
            print(f"[{stage}] complete ({elapsed:.1f}s)", flush=True)
        except Exception as error:
            status.update(failed_stage=stage, error=f"{type(error).__name__}: {error}")
            write_json(output / "run.json", status)
            record(dict(stage=stage, event="failed", error=str(error)))
            raise
    status.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "run.json", status)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
