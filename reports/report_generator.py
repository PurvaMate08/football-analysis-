"""Generate reports and static charts from actual frame-level datasets."""

import json
import math
from pathlib import Path

import pandas as pd

from analytics.eda import summarize_dataset
from config import REPORTS_DIR, ensure_project_directories
from visualization.static_plots import save_correlation_heatmap, save_player_trend


def write_analytical_reports(
    frame_dataset: pd.DataFrame,
    report_name: str,
) -> list[Path]:
    """Write observed-data summary JSON and static chart PNGs."""
    ensure_project_directories()
    safe_name = Path(report_name).stem
    summary_path = REPORTS_DIR / "generated" / f"{safe_name}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            _json_safe(summarize_dataset(frame_dataset)),
            file,
            indent=2,
            allow_nan=False,
        )
    chart_dir = REPORTS_DIR / "charts"
    trend_path = chart_dir / f"{safe_name}_player_trend.png"
    correlation_path = chart_dir / f"{safe_name}_correlation_heatmap.png"
    save_player_trend(frame_dataset, trend_path)
    outputs = [summary_path, trend_path]
    if len(frame_dataset.select_dtypes(include="number").columns) >= 2:
        save_correlation_heatmap(frame_dataset, correlation_path)
        outputs.append(correlation_path)
    return outputs


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
