from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
api = HfApi(token=HF_TOKEN)

repo_id = f"{HF_USERNAME}/superkart-sales-forecast"

# Create the Space (Docker SDK) if it does not already exist
try:
    api.repo_info(repo_id=repo_id, repo_type="space")
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Creating space '{repo_id}'...")
    create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker",
                private=False, token=HF_TOKEN)

# Expose the username to the running app so it can locate the model repo
api.add_space_variable(repo_id=repo_id, key="HF_USERNAME", value=HF_USERNAME)

# Push every deployment file into the Space
api.upload_folder(
    folder_path="superkart_project/deployment",
    repo_id=repo_id,
    repo_type="space",
    path_in_repo="",
)
print(f"Deployment files pushed -> https://huggingface.co/spaces/{repo_id}")
