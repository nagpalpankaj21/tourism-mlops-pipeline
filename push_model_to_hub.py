from huggingface_hub import HfApi
import os

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REPO = "nagpalpankaj21/Tourism_RF_Clean"

api = HfApi()

api.upload_file(
    path_or_fileobj="model/best_model.pkl",
    path_in_repo="best_model.pkl",
    repo_id=MODEL_REPO,
    repo_type="model",
    token=HF_TOKEN
)

print("Model uploaded to HF.")
