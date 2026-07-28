"""Feature engineering utilities."""

import pandas as pd


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready features."""
    return data.copy()

