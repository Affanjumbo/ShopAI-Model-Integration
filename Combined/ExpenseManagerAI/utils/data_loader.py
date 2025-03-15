import pandas as pd

supermarket_data = pd.read_csv("ExpenseManagerAI/datasets/cleaned_supermarket.csv")
ecommerce_data = pd.read_csv("ExpenseManagerAI/datasets/cleaned_ecommerce.csv")

def get_user_spending(user_id):
    user_supermarket = supermarket_data[supermarket_data["User_ID"] == user_id]
    user_ecommerce = ecommerce_data[ecommerce_data["User_ID"] == user_id]
    
    combined_data = pd.concat([user_supermarket, user_ecommerce])
    
    spending_per_category = combined_data.groupby("Category")["Price"].sum().to_dict()
    
    return spending_per_category
