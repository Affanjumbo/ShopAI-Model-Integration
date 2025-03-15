import torch
import torch.nn as nn
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (React, Flutter, Postman, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],
)

# Load dataset
dataset_path = "RecommendationAI/datasets/cleaned_ecommerce.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

df = pd.read_csv(dataset_path)
df = df.rename(columns={"User_ID": "user_id", "Product_ID": "product_id"})

# Create user and product mappings
user_mapping = {user: idx for idx, user in enumerate(df["user_id"].unique())}
product_mapping = {product: idx for idx, product in enumerate(df["product_id"].unique())}
reverse_product_mapping = {idx: product for product, idx in product_mapping.items()}

num_users = len(user_mapping)
num_products = len(product_mapping)

# Define recommendation model
class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_products, embedding_size=50):
        super(RecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.product_embedding = nn.Embedding(num_products, embedding_size)
        self.fc = nn.Linear(embedding_size, 1)

    def forward(self, user_ids, product_ids):
        user_vecs = self.user_embedding(user_ids)
        product_vecs = self.product_embedding(product_ids)
        interaction = user_vecs * product_vecs
        return self.fc(interaction).squeeze()

# Load trained model
model_path = "RecommendationAI/models/recommendation_model.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = RecommendationModel(num_users, num_products)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Request model for recommendations
class RecommendationRequest(BaseModel):
    user_id: str
    top_k: int = 5  # Number of recommendations

# Recommendation endpoint
@app.post("/recommend/")
def get_recommendations(request: RecommendationRequest):
    user_id = request.user_id
    top_k = request.top_k

    if user_id not in user_mapping:
        return {"error": "User ID not found! Showing popular products."}

    user_idx = user_mapping[user_id]
    product_indices = list(product_mapping.values())
    user_tensor = torch.tensor([user_idx] * len(product_indices), dtype=torch.long)
    product_tensor = torch.tensor(product_indices, dtype=torch.long)

    with torch.no_grad():
        scores = model(user_tensor, product_tensor)

    top_indices = torch.argsort(scores, descending=True)[:top_k]
    recommended_products = [reverse_product_mapping[idx.item()] for idx in top_indices]

    return {"user_id": user_id, "recommended_products": recommended_products}

# Root endpoint to check API status
@app.get("/")
def home():
    return {"message": "ShopAI Recommendation API is running!"}