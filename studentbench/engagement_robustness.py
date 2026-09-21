"""Exploratory checks of the engagement/practice associations in Appendix H.

Recompute all 462 recorded diagnostics: partial ranks, available-case fits,
within-section 1st/99th-percentile winsorization, and omitted-model fits. These
check effect direction and stability; their p-values are descriptive only.
"""

from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
import pandas as pd
from scipy import stats
from .engagement import extract, fit_tests, csv_rows, write_json
from . import engagement_models as models
from .journal import Journal


def partial_rank(frame, spec):
    x = stats.rankdata(frame[spec["x"]].to_numpy(float))
    y = stats.rankdata(frame[spec["y"]].to_numpy(float))
    pre = stats.rankdata(frame.pre_pct.to_numpy(float))
    columns = [np.ones(len(frame)), (pre - pre.mean()) / (pre.std(ddof=1) or 1)]
    forms = (
        frame.instrument + ":" + frame.form_order
        if spec["scope"] == "combined"
        else frame.form_order
    )
    for values in [frame.arm_id, forms]:
        for level in sorted(values.unique())[1:]:
            columns.append((values.to_numpy() == level).astype(float))
    design = np.column_stack(columns)
    x -= design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y -= design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return dict(n=len(frame), partial_rank_correlation=float(np.corrcoef(x, y)[0, 1]))


def run(data_dir, output_dir, engagement_dir=None):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engagement_dir = (
        Path(engagement_dir) if engagement_dir else output_dir / "engagement"
    )
    rows = extract(data_dir, engagement_dir)
    main = fit_tests(rows, engagement_dir)
    data = pd.DataFrame(rows)
    complete = data.loc[data.common_primary].copy()
    primary = [dict(r) for r in main["tests"] if r["family"] == "primary_raw"]
    for row, p in zip(
        primary, models.adjustments([r["p_raw"] for r in primary])["holm"]
    ):
        row["primary78_holm"] = p
    signature = hashlib.sha256(
        json.dumps(rows, sort_keys=True, allow_nan=False).encode()
        + Path(__file__).read_bytes()
        + Path(models.__file__).read_bytes()
    ).hexdigest()
    journal = Journal(output_dir / "diagnostics.jsonl")
    diagnostics = []

    def calculate(family, spec, frame, omit=None):
        key = ":".join(
            [family, spec["scope"], spec["x"], spec["y"]] + ([omit] if omit else [])
        )
        cached = journal.get(key, signature)
        if cached is not None:
            diagnostics.append(cached)
            return cached
        sub = (
            frame
            if spec["scope"] == "combined"
            else frame.loc[frame.instrument == spec["scope"]]
        )
        sub = sub.dropna(
            subset=[spec["x"], spec["y"], "pre_pct", "arm_id", "form_order"]
        )
        if omit is not None:
            sub = sub.loc[sub.arm_id != omit]
        fit = (
            partial_rank(sub, spec)
            if family == "diagnostic_partial_rank"
            else models.scaled_fit(sub, spec)
        )
        row = {k: spec[k] for k in ["scope", "x", "y", "edge"]}
        row.update(
            record_id=key,
            family=family,
            omitted_model=omit,
            diagnostic_only=True,
            status="complete",
            **fit,
        )
        journal.save(key, signature, row)
        diagnostics.append(row)
        return row

    for spec in primary:
        calculate("diagnostic_partial_rank", spec, complete)
        calculate("diagnostic_all_available", spec, data)
    clamped = complete.copy().astype(
        {metric: float for metric in models.L + models.E + models.P}
    )
    cuts = []
    for section in ["quant", "verbal"]:
        mask = clamped.instrument == section
        for metric in models.L + models.E + models.P:
            lower, upper = np.quantile(complete.loc[mask, metric], [0.01, 0.99])
            clamped.loc[mask, metric] = np.clip(clamped.loc[mask, metric], lower, upper)
            cuts.append(
                dict(
                    section=section, metric=metric, low=float(lower), high=float(upper)
                )
            )
    write_json(output_dir / "winsor_cutpoints.json", cuts)
    for spec in primary:
        calculate("diagnostic_winsor_1_99", spec, clamped)
    lookup = {(r["scope"], r["x"], r["y"]): r for r in primary}
    candidates = {}
    for scope, latency, engagement, practice in itertools.product(
        models.SCOPES, models.L, models.E, models.P
    ):
        edges = [
            lookup[scope, latency, engagement],
            lookup[scope, engagement, practice],
            lookup[scope, practice, "gain_pp"],
        ]
        if (
            all(r["estimate_per_unit"] * sign > 0 for r, sign in zip(edges, [-1, 1, 1]))
            and max(r["primary78_holm"] for r in edges) < 0.05
        ):
            for edge in edges:
                candidates[edge["record_id"]] = edge
    for spec in candidates.values():
        scope = (
            complete
            if spec["scope"] == "combined"
            else complete.loc[complete.instrument == spec["scope"]]
        )
        for model in sorted(scope.arm_id.unique()):
            calculate("diagnostic_leave_one_model_out", spec, complete, model)
    excluded = {
        r["student_id"]
        for r in csv_rows(
            data_dir / "6_population_demographics/repeat_participant_sessions.csv"
        )
    }
    reduced = complete.loc[~complete.student_id.isin(excluded)]
    repeats = []
    for key, spec in sorted(candidates.items()):
        if spec["scope"] != "combined":
            continue
        cache_key = "repeat:" + key
        value = journal.get(cache_key, signature)
        if value is None:
            full = models.scaled_fit(complete, spec)
            cut = models.scaled_fit(reduced, spec)
            value = dict(
                record_id=key,
                x=spec["x"],
                y=spec["y"],
                original=full,
                exclude_repeat=cut,
                sign_preserved=bool(
                    full["estimate_per_unit"] * cut["estimate_per_unit"] > 0
                ),
            )
            journal.save(cache_key, signature, value)
        repeats.append(value)
    assert len(diagnostics) == 462 and len(repeats) == 7
    result = dict(
        status="complete",
        complete=True,
        diagnostic_fits=diagnostics,
        repeat_participation=repeats,
        diagnostic_family_counts={
            name: sum(r["family"] == name for r in diagnostics)
            for name in sorted({r["family"] for r in diagnostics})
        },
        all_repeat_signs_preserved=all(r["sign_preserved"] for r in repeats),
        interpretation="Post-hoc descriptive robustness; does not change the Holm276 inference family.",
    )
    write_json(output_dir / "results.json", result)
    return result
