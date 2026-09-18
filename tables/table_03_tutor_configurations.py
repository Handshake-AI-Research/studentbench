"""Table 3: recorded tutor endpoints and analyzed assignments."""

from pathlib import Path
from studentbench.table_output import render, cli
from studentbench.data import Dataset, load_sessions
from studentbench.plotting import display_label


def run(data_dir, analysis_dir, output_dir):
    data = Dataset(Path(data_dir))
    frame = load_sessions(Path(data_dir), Path(analysis_dir) / "learning")
    endpoints = {}
    for _, profile in data.profiles():
        if profile["tutoring_arm"] == "ai":
            endpoints.setdefault(profile["ai_model_preset_id"], set()).add(
                profile["ai_model_endpoint_id"]
            )
    rows = []
    for arm in sorted(endpoints, key=lambda arm: display_label(arm).lower()):
        assert len(endpoints[arm]) == 1
        rows.append(
            {
                "AI tutor": display_label(arm),
                "Endpoint": next(iter(endpoints[arm])),
                "Quant": int(((frame.arm_id == arm) & frame.section.eq("quant")).sum()),
                "Verbal": int(
                    ((frame.arm_id == arm) & frame.section.eq("verbal")).sum()
                ),
            }
        )
    return render(
        output_dir,
        "table_03_tutor_configurations",
        rows,
        "Table 3. Tutor configurations and assignments",
        "Endpoints are recorded study settings; assignment counts are recomputed.",
    )


if __name__ == "__main__":
    cli(run)
