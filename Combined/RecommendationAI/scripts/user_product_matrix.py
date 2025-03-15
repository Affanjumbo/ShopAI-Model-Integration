import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load cleaned datasets
ecom_df = pd.read_csv("Combined/RecommendationAI/datasets/cleaned_ecommerce.csv")
supermarket_df = pd.read_csv("RecommendationAI/datasets/cleaned_supermarket.csv")

# 🛒 Step 1: Build User-Product Interaction Matrix
# For e-commerce dataset
ecom_interaction = ecom_df.pivot_table(index="User_ID", columns="Product_ID", values="Final_Price(Rs.)", aggfunc="sum").fillna(0)

# For supermarket dataset
supermarket_interaction = supermarket_df.pivot_table(index="Invoice ID", columns="Product line", values="Total", aggfunc="sum").fillna(0)

# 📈 Step 2: Compute Product Popularity Scores
ecom_popularity = ecom_df.groupby("Product_ID")["User_ID"].count().reset_index()
ecom_popularity.columns = ["Product_ID", "Total_Purchases"]
ecom_popularity = ecom_popularity.sort_values(by="Total_Purchases", ascending=False)

supermarket_popularity = supermarket_df.groupby("Product line")["Invoice ID"].count().reset_index()
supermarket_popularity.columns = ["Product line", "Total_Purchases"]
supermarket_popularity = supermarket_popularity.sort_values(by="Total_Purchases", ascending=False)

# 👥 Step 3: Identify Users with Similar Purchase Behavior
ecom_similarity = cosine_similarity(ecom_interaction)
supermarket_similarity = cosine_similarity(supermarket_interaction)

# Convert similarity matrix to DataFrame
ecom_similarity_df = pd.DataFrame(ecom_similarity, index=ecom_interaction.index, columns=ecom_interaction.index)
supermarket_similarity_df = pd.DataFrame(supermarket_similarity, index=supermarket_interaction.index, columns=supermarket_interaction.index)

# Save matrices
ecom_interaction.to_csv("RecommendationAI/datasets/ecom_user_product_matrix.csv")
supermarket_interaction.to_csv("RecommendationAI/datasets/supermarket_user_product_matrix.csv")

ecom_popularity.to_csv("RecommendationAI/datasets/ecom_product_popularity.csv", index=False)
supermarket_popularity.to_csv("RecommendationAI/datasets/supermarket_product_popularity.csv", index=False)

ecom_similarity_df.to_csv("RecommendationAI/datasets/ecom_user_similarity.csv")
supermarket_similarity_df.to_csv("RecommendationAI/datasets/supermarket_user_similarity.csv")

print("✅ User-Product Matrices, Popularity Scores & User Similarities saved successfully!")
