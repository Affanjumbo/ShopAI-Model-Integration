import pandas as pd  

ecommerce_data = pd.read_csv("ExpenseManagerAI/datasets/cleaned_ecommerce.csv")
supermarket_data = pd.read_csv("ExpenseManagerAI/datasets/final_supermarket.csv")

ecommerce_data["Purchase_Date"] = pd.to_datetime(ecommerce_data["Purchase_Date"], format="%d-%m-%Y")  
supermarket_data["Date"] = pd.to_datetime(supermarket_data["Date"], format="%Y-%m-%d") 

ecommerce_data = ecommerce_data[["User_ID", "Product_ID", "Category", "Final_Price(Rs.)", "Purchase_Date"]]
supermarket_data = supermarket_data[["User_ID", "Product line", "Total", "Date"]]

supermarket_data.rename(columns={"Product line": "Category", "Total": "Final_Price(Rs.)", "Date": "Purchase_Date"}, inplace=True)

final_data = pd.concat([ecommerce_data, supermarket_data], ignore_index=True)

final_data.to_csv("ExpenseManagerAI/datasets/processed_purchases.csv", index=False)
