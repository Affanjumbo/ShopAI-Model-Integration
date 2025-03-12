import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

# Load the dataset
df = pd.read_csv("F:/ShopAI-AI-Models/datasets/cleaned_ecommerce.csv")  # Replace with your actual dataset

# Create unique mappings for users and products
unique_users = df["User_ID"].unique()
unique_products = df["Product_ID"].unique()

# Assign indices
user_mapping = {user: idx for idx, user in enumerate(unique_users)}
product_mapping = {product: idx for idx, product in enumerate(unique_products)}

# Save mappings as pickle files
with open("user_mapping.pkl", "wb") as f:
    pickle.dump(user_mapping, f)

with open("product_mapping.pkl", "wb") as f:
    pickle.dump(product_mapping, f)

print("User and product mappings saved successfully!")



# Load the cleaned datasets
ecom_df = pd.read_csv(r"F:/ShopAI-AI-Models/datasets/cleaned_ecommerce.csv")
supermarket_df = pd.read_csv(r"F:/ShopAI-AI-Models/datasets/cleaned_supermarket.csv")

# Encoding categorical features
label_encoder = LabelEncoder()
ecom_df["Category"] = label_encoder.fit_transform(ecom_df["Category"])
supermarket_df["Payment"] = label_encoder.fit_transform(supermarket_df["Payment"])

# Feature scaling
scaler = MinMaxScaler()
ecom_df[["Final_Price(Rs.)"]] = scaler.fit_transform(ecom_df[["Final_Price(Rs.)"]])
supermarket_df[["Total"]] = scaler.fit_transform(supermarket_df[["Total"]])

# Extract date features
ecom_df["Purchase_Date"] = pd.to_datetime(ecom_df["Purchase_Date"], dayfirst=True, errors='coerce')
ecom_df["Year"] = ecom_df["Purchase_Date"].dt.year
ecom_df["Month"] = ecom_df["Purchase_Date"].dt.month
ecom_df["Day"] = ecom_df["Purchase_Date"].dt.day


supermarket_df["Date"] = pd.to_datetime(supermarket_df["Date"], format="mixed", errors='coerce')
supermarket_df["Year"] = supermarket_df["Date"].dt.year
supermarket_df["Month"] = supermarket_df["Date"].dt.month
supermarket_df["Day"] = supermarket_df["Date"].dt.day

# Train-Test Split
ecom_train, ecom_test = train_test_split(ecom_df, test_size=0.2, random_state=42)
supermarket_train, supermarket_test = train_test_split(supermarket_df, test_size=0.2, random_state=42)

# Save the processed data
ecom_train.to_csv("datasets/ecom_train.csv", index=False)
ecom_test.to_csv("datasets/ecom_test.csv", index=False)
supermarket_train.to_csv("datasets/supermarket_train.csv", index=False)
supermarket_test.to_csv("datasets/supermarket_test.csv", index=False)

print("Data preprocessing complete. Processed datasets saved.")
