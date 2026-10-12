
import unittest
import pandas as pd

from wine_quality import load_data, prepare_data


class TestWineQuality(unittest.TestCase):

    def test_load_data(self):
        data = load_data("WineQT.csv")
        self.assertGreater(len(data), 0)
        self.assertIn("quality", data.columns)

    def test_prepare_data(self):
        data = pd.DataFrame({
            "Id": [1, 2, 3],
            "alcohol": [9.5, 10.0, 11.0],
            "pH": [3.2, 3.3, 3.4],
            "quality": [5, 6, 7]
        })

        X, y = prepare_data(data)

        self.assertEqual(len(X), 3)
        self.assertEqual(len(y), 3)
        self.assertNotIn("quality", X.columns)
        self.assertNotIn("Id", X.columns)

    def test_target_values(self):
        data = load_data("WineQT.csv")
        self.assertTrue(data["quality"].between(0, 10).all())


if __name__ == "__main__":
    unittest.main()
