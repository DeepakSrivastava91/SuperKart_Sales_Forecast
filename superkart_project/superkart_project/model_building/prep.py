# for data manipulation
import pandas as pd
import os
# for splitting
from sklearn.model_selection import train_test_split
# for hugging face
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
api = HfApi(token=HF_TOKEN)

DATASET_REPO = f"{HF_USERNAME}/superkart"
REFERENCE_YEAR = 2009  # latest establishment year in the data

# Load the dataset directly from the Hugging Face dataset space
df = pd.read_csv(f"hf://datasets/{DATASET_REPO}/SuperKart.csv")
print("Dataset loaded from HF:", df.shape)

# ---------------- cleaning & feature engineering ----------------
# 1. Fix the inconsistent category label ('reg' -> 'Regular')
df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})
# 2. Broad category from the product-id prefix (FD/NC/DR)
df["Product_Category"] = df["Product_Id"].str[:2]
# 3. Store age is more informative than the raw establishment year
df["Store_Age"] = REFERENCE_YEAR - df["Store_Establishment_Year"]
# 4. Drop unnecessary columns (unique id + replaced year)
df = df.drop(columns=["Product_Id", "Store_Establishment_Year"])

target = "Product_Store_Sales_Total"

numeric_features = [
    "Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age",
]
categorical_features = [
    "Product_Sugar_Content", "Product_Type", "Store_Id", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Product_Category",
]

X = df[numeric_features + categorical_features]
y = df[target]

# Split into train / test
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload the splits back to the Hugging Face dataset space
for file_path in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print("Uploaded", file_path)
print("Data preparation complete.")
