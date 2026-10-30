# Wine Quality MLOps

An end-to-end MLOps project built from `WineQT.csv`. It trains a wine-quality classifier, validates it with a quality gate, serves predictions through Flask, packages the service with Docker, exposes Prometheus metrics, and includes GitHub Actions and Kubernetes deployment workflows.

## What is included

- Reproducible training with a stratified train/test split
- Saved model, metrics, and feature metadata
- Unit tests and an ML quality gate
- Browser dashboard at `/` plus JSON API endpoints `/health`, `/predict`, and `/metrics`
- Structured request logging and optional Sentry error tracking
- Docker image and Compose stack with Prometheus and Grafana
- GHCR publishing workflow
- Kubernetes Deployment, Service, HorizontalPodAutoscaler, and ConfigMap
- Scheduled data-drift detection and conditional retraining
- Render Blueprint configuration

## Run locally

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\train.py
pytest -q
python -m app
```

Open `http://localhost:5000/` for the prediction dashboard, or `http://localhost:5000/health` to check model readiness. The dashboard submits the 11 wine features to the prediction API:

```powershell
Invoke-RestMethod http://localhost:5000/predict -Method Post `
  -ContentType 'application/json' `
  -Body '{"fixed acidity":7.4,"volatile acidity":0.7,"citric acid":0.0,"residual sugar":1.9,"chlorides":0.076,"free sulfur dioxide":11,"total sulfur dioxide":34,"density":0.9978,"pH":3.51,"sulphates":0.56,"alcohol":9.4}'
```

The response contains the predicted quality score and probability for each possible score.

## Docker

```powershell
docker compose up --build
```

The API is available at `http://localhost:5000`, Prometheus at `http://localhost:9090`, and Grafana at `http://localhost:3000`.

### Monitoring demonstration

After logging into Grafana with the local `admin` account, open **Dashboards > Wine Quality MLOps**. The project automatically provisions a Prometheus data source and dashboard with:

- API availability (`1` means Prometheus can scrape the API)
- Requests per second
- 95th-percentile API latency
- Prediction counts by wine-quality score
- HTTP responses grouped by status code
- Training dataset row count and quality distribution
- Training feature means

Create a few predictions in the API dashboard at `http://localhost:5000`. Within 10–30 seconds, the Grafana panels will show the request and prediction activity. Prometheus is the time-series database; Grafana is the visualization layer.

Prometheus stores dataset summaries, not every raw CSV cell. The raw records remain in [`data/WineQT.csv`](./data/WineQT.csv); Prometheus exposes `wine_dataset_rows`, `wine_dataset_quality_rows`, and `wine_dataset_feature_mean` for monitoring and dashboarding.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `MODEL_PATH` | `artifacts/model.joblib` | Model file used by the API |
| `METRICS_PATH` | `artifacts/metrics.json` | Training metrics file |
| `LOG_LEVEL` | `INFO` | Application log level |
| `SENTRY_DSN` | empty | Optional Sentry DSN |

Never put GitHub passwords or access tokens in this repository. Configure `GHCR_TOKEN`, `KUBE_CONFIG`, and deployment credentials as repository/environment secrets.

## CI/CD workflow

The workflows in `.github/workflows` cover:

1. test and train
2. enforce the model quality gate
3. upload model and metrics as workflow artifacts
4. build and smoke-test the Docker image
5. publish the tested image to GHCR
6. deploy to Kubernetes
7. check drift and retrain when the PSI threshold is exceeded
