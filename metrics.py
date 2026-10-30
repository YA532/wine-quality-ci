from prometheus_client import Counter, Gauge, Histogram

DATASET_ROWS = Gauge("wine_dataset_rows", "Number of rows in the training dataset")
DATASET_QUALITY_ROWS = Gauge(
    "wine_dataset_quality_rows",
    "Number of training rows for each quality score",
    ["quality"],
)
DATASET_FEATURE_MEAN = Gauge(
    "wine_dataset_feature_mean",
    "Mean value of each numeric model feature in the training dataset",
    ["feature"],
)

REQUEST_COUNT = Counter(
    "wine_api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "wine_api_request_latency_seconds", "API request latency", ["endpoint"]
)
PREDICTION_COUNT = Counter(
    "wine_predictions_total", "Total predictions", ["predicted_quality"]
)
