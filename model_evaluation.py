from datasets import load_dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

DATASET_ID = "nagpalpankaj21/Tourism_Clean"

test = load_dataset(DATASET_ID, data_files={"test":"test.csv"})["test"].to_pandas()
model = joblib.load("model/best_model.pkl")

X_test = test.drop(columns=["ProdTaken"])
y_test = test["ProdTaken"]

for col in X_test.columns:
    if X_test[col].dtype == "object":
        X_test[col] = LabelEncoder().fit_transform(X_test[col])

preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))
