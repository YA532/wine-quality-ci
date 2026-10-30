import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "artifacts" / "model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", ROOT / "artifacts" / "metrics.json"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

