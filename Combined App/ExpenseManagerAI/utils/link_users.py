import pandas as pd

ecommerce_data = pd.read_csv("ExpenseManagerAI/datasets/cleaned_ecommerce.csv")
supermarket_data = pd.read_csv("ExpenseManagerAI/datasets/cleaned_supermarket.csv")

if not all(col in ecommerce_data.columns for col in ["User_ID", "Payment_Method", "Final_Price(Rs.)", "Purchase_Date"]):
    raise ValueError("E-commerce dataset is missing required columns!")

if not all(col in supermarket_data.columns for col in ["Payment", "Total", "Date"]):
    raise ValueError("Supermarket dataset is missing required columns!")

ecommerce_data["Purchase_Date"] = pd.to_datetime(ecommerce_data["Purchase_Date"], format="%d-%m-%Y", errors="coerce")
supermarket_data["Date"] = pd.to_datetime(supermarket_data["Date"], format="%d-%m-%Y", errors="coerce")

def match_users():
    user_map = []
    for idx, row in supermarket_data.iterrows():
        potential_matches = ecommerce_data[
            (ecommerce_data["Purchase_Date"] == row["Date"]) &
            (ecommerce_data["Payment_Method"] == row["Payment"])
        ]
        if not potential_matches.empty:
            price_match = potential_matches.iloc[(potential_matches["Final_Price(Rs.)"] - row["Total"]).abs().argsort()[:1]]
            if not price_match.empty:
                matched_user = price_match["User_ID"].values[0]
                user_map.append(matched_user)
            else:
                user_map.append(None)
        else:
            user_map.append(None)
    supermarket_data["User_ID"] = user_map
    supermarket_data.to_csv("ExpenseManagerAI/datasets/final_supermarket.csv", index=False)
    print("Updated supermarket dataset saved with User_ID.")

match_users()
