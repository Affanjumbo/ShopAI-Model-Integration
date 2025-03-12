import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Paths to input and output files
ecom_file = "F:/ShopAI-AI-Models/datasets/cleaned_ecommerce.csv"
supermarket_file = "F:/ShopAI-AI-Models/datasets/cleaned_supermarket.csv"

output_ecom_train = "F:/ShopAI-AI-Models/datasets/ecommerce_train.csv"
output_ecom_test = "F:/ShopAI-AI-Models/datasets/ecommerce_test.csv"

output_supermarket_train = "F:/ShopAI-AI-Models/datasets/supermarket_train.csv"
output_supermarket_test = "F:/ShopAI-AI-Models/datasets/supermarket_test.csv"

# Load cleaned datasets
ecom_df = pd.read_csv(ecom_file)
supermarket_df = pd.read_csv(supermarket_file)

# Ensure correct data types for dates
ecom_df["Purchase_Date"] = pd.to_datetime(ecom_df["Purchase_Date"], format="%d-%m-%Y", dayfirst=True)
supermarket_df["Date"] = pd.to_datetime(supermarket_df["Date"], format="%d-%m-%Y", dayfirst=True)

# Extract Year, Month, and Day
ecom_df["Year"] = ecom_df["Purchase_Date"].dt.year
ecom_df["Month"] = ecom_df["Purchase_Date"].dt.month
ecom_df["Day"] = ecom_df["Purchase_Date"].dt.day

supermarket_df["Year"] = supermarket_df["Date"].dt.year
supermarket_df["Month"] = supermarket_df["Date"].dt.month
supermarket_df["Day"] = supermarket_df["Date"].dt.day

# Map payment methods to your project's versions
payment_mapping = {
    "Credit Card": "MasterCard",
    "Debit Card": "Visa",
    "PayPal": "Easypaisa",
    "Net Banking": "JazzCash",
    "Cash": "COD"
}

ecom_df["Payment_Method"] = ecom_df["Payment_Method"].map(payment_mapping).fillna(ecom_df["Payment_Method"])
supermarket_df["Payment"] = supermarket_df["Payment"].map(payment_mapping).fillna(supermarket_df["Payment"])

# Replace foreign city names with Pakistani cities
pakistani_cities = ["Karachi", "Lahore", "Islamabad", "Faisalabad", "Rawalpindi", "Multan", "Peshawar", "Quetta"]
supermarket_df["City"] = supermarket_df["City"].apply(lambda x: pakistani_cities[hash(x) % len(pakistani_cities)])

# Ensure currency format is Pakistani Rupees (Rs.)
ecom_df["Price (Rs.)"] = ecom_df["Price (Rs.)"].apply(lambda x: f"Rs. {x}")
ecom_df["Final_Price(Rs.)"] = ecom_df["Final_Price(Rs.)"].apply(lambda x: f"Rs. {x}")
supermarket_df["Unit price"] = supermarket_df["Unit price"].apply(lambda x: f"Rs. {x}")
supermarket_df["Total"] = supermarket_df["Total"].apply(lambda x: f"Rs. {x}")

# Generate New Features

## E-Commerce Dataset
ecom_df["Recency"] = (pd.to_datetime("2025-03-09") - ecom_df["Purchase_Date"]).dt.days
ecom_frequency = ecom_df.groupby("User_ID").size().reset_index(name="Frequency")
ecom_monetary = ecom_df.groupby("User_ID")["Final_Price(Rs.)"].apply(lambda x: sum(float(str(i).replace("Rs. ", "")) for i in x)).reset_index()
ecom_monetary.rename(columns={"Final_Price(Rs.)": "Monetary_Value"}, inplace=True)

# Merge features
ecom_features = ecom_df.merge(ecom_frequency, on="User_ID").merge(ecom_monetary, on="User_ID")

## Supermarket Dataset
supermarket_df["Recency"] = (pd.to_datetime("2025-03-09") - supermarket_df["Date"]).dt.days
supermarket_frequency = supermarket_df.groupby("Invoice ID").size().reset_index(name="Frequency")
supermarket_monetary = supermarket_df.groupby("Invoice ID")["Total"].apply(lambda x: sum(float(str(i).replace("Rs. ", "")) for i in x)).reset_index()
supermarket_monetary.rename(columns={"Total": "Monetary_Value"}, inplace=True)

# Merge features
supermarket_features = supermarket_df.merge(supermarket_frequency, on="Invoice ID").merge(supermarket_monetary, on="Invoice ID")

# Convert Categorical Data to Numerical (Only for ML Models, Not Final CSV)
ecom_label_cols = ["Category"]
supermarket_label_cols = ["Customer type", "Gender", "Product line"]

for col in ecom_label_cols:
    le = LabelEncoder()
    ecom_features[col] = le.fit_transform(ecom_features[col])

for col in supermarket_label_cols:
    le = LabelEncoder()
    supermarket_features[col] = le.fit_transform(supermarket_features[col])

# Split into Train (80%) and Test (20%)
ecom_train = ecom_features.sample(frac=0.8, random_state=42)
ecom_test = ecom_features.drop(ecom_train.index)

supermarket_train = supermarket_features.sample(frac=0.8, random_state=42)
supermarket_test = supermarket_features.drop(supermarket_train.index)

# Save engineered datasets
ecom_train.to_csv(output_ecom_train, index=False)
ecom_test.to_csv(output_ecom_test, index=False)

supermarket_train.to_csv(output_supermarket_train, index=False)
supermarket_test.to_csv(output_supermarket_test, index=False)

print("✅ Feature Engineering Completed! Train and Test files saved in 'datasets'")
