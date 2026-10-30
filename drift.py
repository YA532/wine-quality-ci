import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import ROOT
from app.model import FEATURE_COLUMNS

REFERENCE = ROOT / "data" / "WineQT.csv"
CURRENT = Path(__file__).resolve().parent / "current_data.csv"
REPORT_DIR = ROOT / "drift_reports"
PSI_THRESHOLD = 0.2


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 2:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_ratio = np.maximum(expected_counts / len(expected), 1e-6)
    actual_ratio = np.maximum(actual_counts / len(actual), 1e-6)
    return float(np.sum((actual_ratio - expected_ratio) * np.log(actual_ratio / expected_ratio)))


def main() -> int:
    if not CURRENT.exists():
        print(f"No current data found at {CURRENT}; drift check skipped")
        return 0
    reference = pd.read_csv(REFERENCE)
    current = pd.read_csv(CURRENT)
    scores = {feature: psi(reference[feature].to_numpy(), current[feature].to_numpy()) for feature in FEATURE_COLUMNS}
    report = {"features": scores, "max_psi": max(scores.values()), "threshold": PSI_THRESHOLD}
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["max_psi"] >= PSI_THRESHOLD else 0


if __name__ == "__main__":
    raise SystemExit(main())
