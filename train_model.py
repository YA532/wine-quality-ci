
import json
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = "WineQT.csv"


def train_model():
    print("Loading WineQT dataset...")

    data = pd.read_csv(DATA_FILE)
    data = data.dropna()

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))

    # Use wine measurements to predict quality.
    X = data.drop(columns=["quality", "Id"], errors="ignore")
    y = data["quality"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    print("Training wine quality model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=sorted(y.unique())
    )

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(float(accuracy), 4))
    print("Quality labels:", sorted(y.unique()))
    print("Confusion Matrix:")
    print(matrix)

    joblib.dump(model, "wine_quality_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": int(len(X_train)),
        "testing_records": int(len(X_test)),
        "quality_labels": [int(value) for value in sorted(y.unique())],
        "features": list(X.columns)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model saved as wine_quality_model.pkl")
    print("Metrics saved as metrics.json")

    return accuracy


if __name__ == "__main__":
    train_model()
