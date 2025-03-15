import pandas as pd

# Load dataset
data = pd.read_csv("ExpenseManagerAI/datasets/augmented_ecommerce_data.csv")

# Convert purchase date to datetime format
data["Purchase_Date"] = pd.to_datetime(data["Purchase_Date"])
data["Month"] = data["Purchase_Date"].dt.to_period("M")

# Calculate total spending per month per user
monthly_spending = data.groupby(["User_ID", "Month"])["Final_Price(Rs.)"].sum().reset_index()

# Calculate category-wise spending per user
category_spending = data.groupby(["User_ID", "Category"])["Final_Price(Rs.)"].sum().reset_index()

# Calculate purchase frequency per category per user
purchase_frequency = data.groupby(["User_ID", "Category"])["Product_ID"].count().reset_index()
purchase_frequency.rename(columns={"Product_ID": "Purchase_Frequency"}, inplace=True)

# Calculate average monthly spending per user
avg_monthly_spending = monthly_spending.groupby("User_ID")["Final_Price(Rs.)"].mean().reset_index()
avg_monthly_spending.rename(columns={"Final_Price(Rs.)": "Avg_Monthly_Spending"}, inplace=True)

# Calculate total discounted purchases per user
discounted_purchases = data[data["Discount (%)"] > 0].groupby("User_ID")["Product_ID"].count().reset_index()
discounted_purchases.rename(columns={"Product_ID": "Discounted_Purchases"}, inplace=True)

# Merge extracted features
features = avg_monthly_spending.merge(discounted_purchases, on="User_ID", how="left")
features["Discounted_Purchases"] = features["Discounted_Purchases"].fillna(0)

# Save the processed user features
features.to_csv("ExpenseManagerAI/datasets/user_features.csv", index=False)
