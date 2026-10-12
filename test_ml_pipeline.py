
import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("WineQT.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("wine_quality_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        data = pd.read_csv("WineQT.csv").dropna()
        X = data.drop(columns=["quality", "Id"], errors="ignore")

        model = joblib.load("wine_quality_model.pkl")
        prediction = model.predict(X.iloc[[0]])[0]

        valid_quality_values = set(data["quality"].astype(int).unique())
        self.assertEqual(int(prediction), -999)

    def test_metrics_record_counts(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreater(metrics["training_records"], 0)
        self.assertGreater(metrics["testing_records"], 0)


if __name__ == "__main__":
    unittest.main()
