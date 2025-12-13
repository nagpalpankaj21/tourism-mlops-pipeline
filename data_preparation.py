from datasets import load_dataset
from huggingface_hub import HfApi
import pandas as pd
from sklearn.model_selection import train_test_split
import os

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_ID = "nagpalpankaj21/Tourism_Clean"

df = load_dataset("nagpalpankaj21/Tourism")["train"].to_pandas()

df = df.drop(columns=["Unnamed: 0"], errors="ignore")
df = df.drop_duplicates()

train, test = train_test_split(df, test_size=0.2, random_state=42)

os.makedirs("processed_data", exist_ok=True)
train.to_csv("processed_data/train.csv", index=False)
test.to_csv("processed_data/test.csv", index=False)

api = HfApi()

api.upload_file(
    path_or_fileobj="processed_data/train.csv",
    path_in_repo="train.csv",
    repo_id=DATASET_ID,
    repo_type="dataset",
    token=HF_TOKEN
)

api.upload_file(
    path_or_fileobj="processed_data/test.csv",
    path_in_repo="test.csv",
    repo_id=DATASET_ID,
    repo_type="dataset",
    token=HF_TOKEN
)

print("Train/Test uploaded.")
