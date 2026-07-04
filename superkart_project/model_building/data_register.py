from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Credentials are read from the environment (set as GitHub Actions secrets in CI)
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")

repo_id = f"{HF_USERNAME}/superkart"
repo_type = "dataset"

api = HfApi(token=HF_TOKEN)

# Create the dataset repo if it does not already exist
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=HF_TOKEN)
    print(f"Dataset '{repo_id}' created.")

# Upload the raw data folder
api.upload_folder(
    folder_path="superkart_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"Raw data registered -> https://huggingface.co/datasets/{repo_id}")
