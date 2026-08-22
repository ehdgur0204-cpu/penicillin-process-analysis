"""Data loading and preprocessing utilities."""

from pathlib import Path

import pandas as pd


def load_data(path: str | Path, **kwargs: object) -> pd.DataFrame:
    """Load a CSV file."""
    return pd.read_csv(Path(path), **kwargs)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a basic cleaned copy of the input data."""
    return data.drop_duplicates().reset_index(drop=True).copy()

