import logging
import os
import time
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template_string, request
import pandas as pd
from prometheus_client import generate_latest

from .config import LOG_LEVEL
from .metrics import (
    DATASET_FEATURE_MEAN,
    DATASET_QUALITY_ROWS,
    DATASET_ROWS,
    PREDICTION_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from .model import FEATURE_COLUMNS, ModelNotReadyError, load_model, prepare_features

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)
DATASET_PATH = os.getenv(
    "DATASET_PATH", str(Path(__file__).resolve().parents[1] / "data" / "WineQT.csv")
)

DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wine Quality Dashboard</title>
  <style>
    :root { color-scheme: dark; --bg: #0f172a; --panel: #17233b; --muted: #9fb0ca; --accent: #61dafb; --good: #43d17a; }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; font: 15px/1.5 Inter, Segoe UI, sans-serif; color: #eef5ff; background: radial-gradient(circle at top right, #243b67, var(--bg) 45%); }
    .shell { max-width: 1180px; margin: auto; padding: 34px 22px 50px; }
    header { display: flex; justify-content: space-between; gap: 20px; align-items: center; margin-bottom: 28px; }
    h1 { margin: 0; font-size: clamp(28px, 5vw, 46px); letter-spacing: -1px; }
    h2 { margin-top: 0; }
    .subtitle, label, .hint { color: var(--muted); }
    .badge { border: 1px solid #2e9c68; color: #8df0b2; border-radius: 999px; padding: 8px 13px; white-space: nowrap; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(280px, .8fr); gap: 20px; }
    .panel { background: rgba(23, 35, 59, .9); border: 1px solid #2c4168; border-radius: 18px; padding: 24px; box-shadow: 0 16px 40px rgba(0,0,0,.2); }
    .fields { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 15px; }
    label { display: flex; flex-direction: column; gap: 6px; font-size: 13px; }
    input { width: 100%; color: #f5f9ff; background: #101a2d; border: 1px solid #385174; border-radius: 9px; padding: 11px; font-size: 15px; }
    input:focus { outline: 2px solid var(--accent); border-color: transparent; }
    button { border: 0; border-radius: 10px; padding: 13px 20px; margin-top: 20px; color: #06121e; background: var(--accent); font-weight: 700; cursor: pointer; }
    button:disabled { opacity: .6; cursor: wait; }
    .result { min-height: 150px; display: grid; place-items: center; text-align: center; border: 1px dashed #47628e; border-radius: 14px; padding: 20px; }
    .score { font-size: 72px; line-height: 1; color: #ffd166; font-weight: 800; }
    .error { color: #ff9a9a; }
    .stats { display: grid; gap: 12px; margin-top: 20px; }
    .stat { display: flex; justify-content: space-between; gap: 12px; padding: 13px; border-radius: 10px; background: #101a2d; }
    .stat strong { color: var(--good); }
    .wide { grid-column: 1 / -1; }
    a { color: var(--accent); }
    @media (max-width: 760px) { header, .grid { display: block; } header .badge { display: inline-block; margin-top: 15px; } .panel { margin-bottom: 20px; } }
    @media (max-width: 480px) { .fields { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div><div class="subtitle">MLOps • Wine Quality</div><h1>Prediction Dashboard</h1><p class="subtitle">Estimate wine quality from its laboratory measurements.</p></div>
      <div class="badge" id="status">● Checking model...</div>
    </header>
    <section class="grid">
      <div class="panel">
        <h2>Wine measurements</h2>
        <p class="hint">Enter the 11 features used by the trained Random Forest model.</p>
        <form id="prediction-form">
          <div class="fields">
            {% for name, value in fields %}
            <label>{{ name }}<input name="{{ name }}" type="number" step="any" value="{{ value }}" required></label>
            {% endfor %}
          </div>
          <button id="submit" type="submit">Predict wine quality</button>
        </form>
      </div>
      <div class="panel">
        <h2>Prediction</h2>
        <div class="result" id="result"><span class="hint">Submit the measurements to see the predicted quality score.</span></div>
        <div class="stats">
          <div class="stat"><span>Model status</span><strong id="model-status">Loading</strong></div>
          <div class="stat"><span>API health</span><strong id="api-status">Checking</strong></div>
        </div>
      </div>
      <div class="panel wide">
        <strong>Operations</strong>
        <p class="hint">Metrics are available at <a href="/metrics">/metrics</a>. Monitoring is configured for Prometheus and Grafana.</p>
      </div>
    </section>
  </main>
  <script>
    const form = document.getElementById("prediction-form");
    const result = document.getElementById("result");
    const button = document.getElementById("submit");
    async function checkHealth() {
      try {
        const response = await fetch("/health");
        const data = await response.json();
        document.getElementById("status").textContent = data.model_loaded ? "● Model online" : "● Model unavailable";
        document.getElementById("model-status").textContent = data.model_loaded ? "Ready" : "Unavailable";
        document.getElementById("api-status").textContent = response.ok ? "Healthy" : "Error";
      } catch (error) {
        document.getElementById("status").textContent = "● API offline";
        document.getElementById("model-status").textContent = "Unavailable";
        document.getElementById("api-status").textContent = "Offline";
      }
    }
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      result.innerHTML = '<span class="hint">Calculating...</span>';
      const payload = Object.fromEntries(new FormData(form).entries());
      for (const key of Object.keys(payload)) payload[key] = Number(payload[key]);
      try {
        const response = await fetch("/predict", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload) });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Prediction failed");
        const probabilities = Object.entries(data.probabilities).map(([label, probability]) => `${label}: ${(probability * 100).toFixed(1)}%`).join(" · ");
        result.innerHTML = `<div><div class="score">${data.predicted_quality}</div><div>Predicted quality score</div><p class="hint">${probabilities}</p></div>`;
      } catch (error) {
        result.innerHTML = `<span class="error">${error.message}</span>`;
      } finally {
        button.disabled = false;
        checkHealth();
      }
    });
    checkHealth();
  </script>
</body>
</html>
"""


def create_app(model: Any | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MODEL"] = model
    try:
        dataset = pd.read_csv(DATASET_PATH)
        DATASET_ROWS.set(len(dataset))
        for quality, count in dataset["quality"].value_counts().items():
            DATASET_QUALITY_ROWS.labels(str(quality)).set(int(count))
        for feature in FEATURE_COLUMNS:
            DATASET_FEATURE_MEAN.labels(feature).set(float(dataset[feature].mean()))
    except (FileNotFoundError, KeyError, pd.errors.ParserError) as exc:
        logger.error("dataset metrics unavailable: %s", exc)
    if os.getenv("SENTRY_DSN"):
        import sentry_sdk
        from sentry_sdk.integrations.flask import FlaskIntegration

        sentry_sdk.init(dsn=os.environ["SENTRY_DSN"], integrations=[FlaskIntegration()])

    @app.before_request
    def start_timer() -> None:
        request.environ["wine.start_time"] = time.perf_counter()

    @app.after_request
    def record_request(response):
        endpoint = request.path
        elapsed = time.perf_counter() - request.environ.get(
            "wine.start_time", time.perf_counter()
        )
        REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
        REQUEST_LATENCY.labels(endpoint).observe(elapsed)
        logger.info(
            "request method=%s path=%s status=%s duration_ms=%.2f",
            request.method,
            endpoint,
            response.status_code,
            elapsed * 1000,
        )
        return response

    @app.get("/health")
    def health():
        try:
            app.config["MODEL"] = app.config["MODEL"] or load_model()
            return jsonify({"status": "ok", "model_loaded": True})
        except ModelNotReadyError:
            return jsonify({"status": "error", "model_loaded": False}), 503

    @app.get("/")
    def index():
        fields = [
            ("fixed acidity", 7.4),
            ("volatile acidity", 0.7),
            ("citric acid", 0.0),
            ("residual sugar", 1.9),
            ("chlorides", 0.076),
            ("free sulfur dioxide", 11.0),
            ("total sulfur dioxide", 34.0),
            ("density", 0.9978),
            ("pH", 3.51),
            ("sulphates", 0.56),
            ("alcohol", 9.4),
        ]
        return render_template_string(
            DASHBOARD_HTML,
            fields=fields,
        )

    @app.get("/metrics")
    def metrics():
        return generate_latest(), 200, {"Content-Type": "text/plain; version=0.0.4"}

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object"}), 400
        try:
            app.config["MODEL"] = app.config["MODEL"] or load_model()
            features = prepare_features(payload)
            model = app.config["MODEL"]
            prediction = int(model.predict(features)[0])
            probabilities = model.predict_proba(features)[0]
            classes = [int(value) for value in model.classes_]
            PREDICTION_COUNT.labels(str(prediction)).inc()
            return jsonify(
                {
                    "predicted_quality": prediction,
                    "probabilities": {
                        str(label): round(float(probability), 6)
                        for label, probability in zip(classes, probabilities)
                    },
                }
            )
        except ModelNotReadyError as exc:
            logger.error("model unavailable: %s", exc)
            return jsonify({"error": str(exc)}), 503
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception:
            logger.exception("prediction failed")
            return jsonify({"error": "Prediction failed"}), 500

    return app
