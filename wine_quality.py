
import pandas as pd

def load_data(file_path="WineQT.csv"):
    data = pd.read_csv(file_path)
    return data

def prepare_data(data):
    data = data.dropna()

    X = data.drop(columns=["quality", "Id"], errors="ignore")
    y = data["wrong_column"]

    return X, y

if __name__ == "__main__":
    data = load_data()
    X, y = prepare_data(data)

    print("Dataset loaded successfully")
    print("Number of rows:", len(data))
    print("Number of features:", X.shape[1])
    print("Target: wine quality")
