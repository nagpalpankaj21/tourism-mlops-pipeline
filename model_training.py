from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib, os

DATASET_ID = "nagpalpankaj21/Tourism_Clean"

train = load_dataset(DATASET_ID, data_files={"train":"train.csv"})["train"].to_pandas()

X = train.drop(columns=["ProdTaken"])
y = train["ProdTaken"]

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = LabelEncoder().fit_transform(X[col])

grid = GridSearchCV(
    RandomForestClassifier(),
    {"n_estimators":[100,150], "max_depth":[5,10,20]},
    cv=3, n_jobs=-1
)

grid.fit(X, y)
best = grid.best_estimator_

os.makedirs("model", exist_ok=True)
joblib.dump(best, "model/best_model.pkl")

print("Model trained.")




