from scripts.train import train


def test_training_produces_quality_gate_metrics():
    metrics = train()
    assert metrics["test_rows"] > 0
    assert metrics["accuracy"] >= metrics["quality_gate_accuracy"]

