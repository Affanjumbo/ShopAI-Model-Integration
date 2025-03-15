import pandas as pd
import random
from datetime import timedelta

file_path = "ExpenseManagerAI/datasets/cleaned_ecommerce.csv"
data = pd.read_csv(file_path)

data = data[data["User_ID"].str.extract(r'U(\d+)')[0].astype(int).between(200, 500)]

data["Purchase_Date"] = pd.to_datetime(data["Purchase_Date"], format="%d-%m-%Y")

def generate_purchases(user_id, num_purchases=5):
    sampled_products = data.sample(n=min(num_purchases, len(data)), replace=True)
    sampled_products["User_ID"] = user_id
    sampled_products["Purchase_Date"] += pd.to_timedelta(
        [random.randint(1, 365) for _ in range(len(sampled_products))], unit="D"
    )
    return sampled_products

augmented_data = pd.concat([generate_purchases(uid, random.randint(5, 20)) for uid in data["User_ID"].unique()])

augmented_data.to_csv("ExpenseManagerAI/datasets/augmented_ecommerce_data.csv", index=False)
print("Augmentation is done")