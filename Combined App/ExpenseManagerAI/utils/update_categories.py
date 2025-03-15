import pandas as pd

# Load dataset
data = pd.read_csv("ExpenseManagerAI/datasets/cleaned_ecommerce.csv")

# Category mapping
category_map = {
    "Groceries": "Monthly Groceries",
    "Personal Care": "Monthly Personal Care Items",
    "Electronics": "Occasional Gadget Purchases",
    "Clothing": "Monthly Clothing & Accessories",
    "Household": "Monthly Household Essentials",
    "Books & Stationery": "Monthly Office & Study Supplies",
}

# Apply mapping
data["Category"] = data["Category"].map(category_map).fillna(data["Category"])

# Save updated dataset
data.to_csv("ExpenseManagerAI/datasets/ecommerce_data_updated.csv", index=False)
