from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import MODEL_PATH

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


class ModelNotReadyError(RuntimeError):
    """Raised when a prediction is requested before a model is trained."""


def load_model(path: Path = MODEL_PATH) -> Any:
    if not path.exists():
        raise ModelNotReadyError(
            f"Model artifact not found at {path}. Run `python scripts/train.py` first."
        )
    return joblib.load(path)


def prepare_features(payload: dict[str, Any]) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in payload]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    values = {}
    for column in FEATURE_COLUMNS:
        try:
            values[column] = float(payload[column])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Feature '{column}' must be numeric") from exc
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)

