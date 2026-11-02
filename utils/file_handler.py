"""File validation and safe, consistent tabular input/output helpers."""

from pathlib import Path
from typing import Iterable

import pandas as pd


def validate_file(path: str | Path, suffixes: Iterable[str] | None = None) -> Path:
    """Validate that a path exists as a file and, optionally, has an accepted suffix."""
    file_path = Path(path).expanduser()
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if suffixes and file_path.suffix.lower() not in {s.lower() for s in suffixes}:
        allowed = ", ".join(suffixes)
        raise ValueError(f"Unsupported file type '{file_path.suffix}'. Expected: {allowed}")
    return file_path


def read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV and surface a useful error for empty or malformed inputs."""
    file_path = validate_file(path, (".csv",))
    try:
        frame = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV has no header or data: {file_path}") from exc
    if frame.empty:
        raise ValueError(f"CSV contains no data rows: {file_path}")
    return frame


def write_csv(frame: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Write a DataFrame, creating its parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=index)
    return output
