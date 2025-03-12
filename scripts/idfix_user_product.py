import pandas as pd
import os

# ✅ Define dataset directory
dataset_dir = "F:/ShopAI-AI-Models/datasets/"

# ✅ List of all dataset files to update
dataset_files = [
    "cleaned_ecommerce.csv",
    "cleaned_supermarket.csv",
    "ecommerce_dataset.csv",
    "ecommerce_test.csv",
    "ecommerce_train.csv",
    "ecom_product_popularity.csv",
    "ecom_user_product_matrix.csv",
    "ecom_user_similarity.csv",
    "supermarket_product_popularity.csv",
    "supermarket_sales.csv",
    "supermarket_test.csv",
    "supermarket_train.csv",
    "supermarket_user_product_matrix.csv",
    "supermarket_user_similarity.csv",
]

# ✅ Collect all unique User_IDs and Product_IDs from all files
all_users = set()
all_products = set()

for file in dataset_files:
    file_path = os.path.join(dataset_dir, file)
    df = pd.read_csv(file_path)
    
    if "User_ID" in df.columns:
        all_users.update(df["User_ID"].unique())
    
    if "Product_ID" in df.columns:
        all_products.update(df["Product_ID"].unique())

# ✅ Create new mappings
user_mapping = {old_id: f"U{i+1}" for i, old_id in enumerate(sorted(all_users))}
product_mapping = {old_id: f"P{i+1}" for i, old_id in enumerate(sorted(all_products))}

# ✅ Update all files
for file in dataset_files:
    file_path = os.path.join(dataset_dir, file)
    df = pd.read_csv(file_path)

    if "User_ID" in df.columns:
        df["User_ID"] = df["User_ID"].map(user_mapping)

    if "Product_ID" in df.columns:
        df["Product_ID"] = df["Product_ID"].map(product_mapping)

    df.to_csv(file_path, index=False)  # ✅ Save updated file

print("✅ All User_IDs and Product_IDs have been successfully updated across all datasets!")
