"""Small SQLite persistence helpers for generated analysis datasets."""

import sqlite3
from pathlib import Path

import pandas as pd

from config import DATABASE_PATH


def save_dataframe(
    frame: pd.DataFrame,
    table_name: str,
    database_path: str | Path = DATABASE_PATH,
) -> None:
    """Replace a table with a DataFrame after validating the SQL identifier."""
    if not table_name.replace("_", "").isalnum():
        raise ValueError("Table names may contain only letters, numbers, and underscores.")
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        frame.to_sql(table_name, connection, if_exists="replace", index=False)


def load_dataframe(
    table_name: str,
    database_path: str | Path = DATABASE_PATH,
) -> pd.DataFrame:
    """Load a persisted table from SQLite."""
    if not table_name.replace("_", "").isalnum():
        raise ValueError("Table names may contain only letters, numbers, and underscores.")
    path = Path(database_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite database does not exist: {path}")
    with sqlite3.connect(path) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        if exists is None:
            raise ValueError(f"Table '{table_name}' does not exist in {path}.")
        return pd.read_sql_query(f'SELECT * FROM "{table_name}"', connection)
