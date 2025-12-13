from huggingface_hub import HfApi, create_repo
import os

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_ID = "nagpalpankaj21/Tourism_Clean"

api = HfApi()

try:
    api.repo_info(DATASET_ID, repo_type="dataset")
    print("Dataset exists.")
except:
    create_repo(DATASET_ID, repo_type="dataset", private=False, token=HF_TOKEN)
    print("Dataset created.")
