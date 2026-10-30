from unittest.mock import Mock

from app.api import create_app


def test_health_and_prediction():
    model = Mock()
    model.predict.return_value = [6]
    model.predict_proba.return_value = [[0.1, 0.9]]
    model.classes_ = [5, 6]
    client = create_app(model).test_client()

    index = client.get("/")
    assert index.status_code == 200
    assert "Wine Quality" in index.get_data(as_text=True)
    assert client.get("/health").status_code == 200
    response = client.post(
        "/predict",
        json={
            "fixed acidity": 7.4,
            "volatile acidity": 0.7,
            "citric acid": 0.0,
            "residual sugar": 1.9,
            "chlorides": 0.076,
            "free sulfur dioxide": 11,
            "total sulfur dioxide": 34,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4,
        },
    )
    assert response.status_code == 200
    assert response.get_json()["predicted_quality"] == 6


def test_prediction_rejects_missing_feature():
    model = Mock()
    client = create_app(model).test_client()
    response = client.post("/predict", json={"alcohol": 9.4})
    assert response.status_code == 400
    assert "Missing features" in response.get_json()["error"]
