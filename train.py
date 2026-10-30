import json
from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import ROOT
from app.model import FEATURE_COLUMNS

DATA_PATH = ROOT / "data" / "WineQT.csv"
ARTIFACTS = ROOT / "artifacts"
QUALITY_GATE_ACCURACY = 0.55


def train() -> dict:
    data = pd.read_csv(DATA_PATH)
    if data.empty or not set(FEATURE_COLUMNS + ["quality"]).issubset(data.columns):
        raise ValueError("Dataset is empty or does not contain the expected schema")
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS],
        data["quality"],
        test_size=0.2,
        random_state=42,
        stratify=data["quality"],
    )
    model = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted")),
        "test_rows": len(y_test),
        "features": FEATURE_COLUMNS,
        "quality_gate_accuracy": QUALITY_GATE_ACCURACY,
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS / "model.joblib")
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ARTIFACTS / "feature_metadata.json").write_text(
        json.dumps({"features": FEATURE_COLUMNS}, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
    if result["accuracy"] < QUALITY_GATE_ACCURACY:
        raise SystemExit("ML quality gate failed: accuracy is below 0.55")
