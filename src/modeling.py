"""Model training and evaluation utilities."""

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_model(
    model: BaseEstimator,
    features: Any,
    target: Any,
) -> BaseEstimator:
    """Fit and return a scikit-learn compatible estimator."""
    return model.fit(features, target)


def evaluate_regression(
    model: BaseEstimator,
    features: Any,
    target: Any,
) -> dict[str, float]:
    """Calculate common regression metrics."""
    predictions = model.predict(features)
    return {
        "mae": float(mean_absolute_error(target, predictions)),
        "rmse": float(mean_squared_error(target, predictions) ** 0.5),
        "r2": float(r2_score(target, predictions)),
    }

