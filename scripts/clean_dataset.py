import pandas as pd

# Load datasets
ecom_df = pd.read_csv("F:/ShopAI-AI-Models/datasets/ecommerce_dataset.csv")
supermarket_df = pd.read_csv("F:/ShopAI-AI-Models/datasets/supermarket_sales.csv")

# **1. Convert USA City Names to Pakistani Cities**
pakistani_cities = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad"]
supermarket_df["City"] = supermarket_df["City"].apply(lambda x: pakistani_cities[hash(x) % len(pakistani_cities)])

# **2. Update Payment Methods**
payment_mapping = {
    "Net Banking": "Easypaisa",
    "Credit Card": "UBL Omni",
    "UPI": "JazzCash",
    "Ewallet": "Easypaisa",
    "Cash": "Cash on Delivery"
}
ecom_df["Payment_Method"] = ecom_df["Payment_Method"].replace(payment_mapping)
supermarket_df["Payment"] = supermarket_df["Payment"].replace(payment_mapping)

# **3. Standardize Date Format (Fix for Mixed Formats)**
ecom_df["Purchase_Date"] = pd.to_datetime(ecom_df["Purchase_Date"], format='mixed', dayfirst=True).dt.strftime("%d-%m-%Y")
supermarket_df["Date"] = pd.to_datetime(supermarket_df["Date"], format='mixed', dayfirst=True).dt.strftime("%d-%m-%Y")

# **4. Drop Unnecessary Columns**
supermarket_df.drop(columns=["Branch"], inplace=True)  # Remove 'Branch' column

# **Save the cleaned datasets**
ecom_df.to_csv("F:/ShopAI-AI-Models/datasets/cleaned_ecommerce.csv", index=False)
supermarket_df.to_csv("F:/ShopAI-AI-Models/datasets/cleaned_supermarket.csv", index=False)

print("Datasets cleaned and saved successfully!")
