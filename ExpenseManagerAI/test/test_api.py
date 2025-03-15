import pandas as pd

supermarket_data = pd.read_csv("ExpenseManagerAI/datasets/cleaned_supermarket.csv")
ecommerce_data = pd.read_csv("ExpenseManagerAI/datasets/cleaned_ecommerce.csv")

print("Dataset Columns:", supermarket_data.columns.tolist())
