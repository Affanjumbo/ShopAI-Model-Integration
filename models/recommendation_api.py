import torch
import torch.nn as nn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# FastAPI Setup
app = FastAPI()

# Enable CORS (Fixes OPTIONS request issues)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains (change this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow GET, POST, etc.
    allow_headers=["*"],
)

# Load Dataset for Product Information
dataset_path = "datasets/cleaned_ecommerce.csv"
if not os.path.exists(dataset_path):
    raise HTTPException(status_code=500, detail="Dataset file is missing!")

df = pd.read_csv(dataset_path)
df = df.rename(columns={"User_ID": "user_id", "Product_ID": "product_id"})

# Map User and Product IDs (Now using string-based IDs)
user_mapping = {user: idx for idx, user in enumerate(df["user_id"].unique())}
product_mapping = {product: idx for idx, product in enumerate(df["product_id"].unique())}
reverse_user_mapping = {idx: user for user, idx in user_mapping.items()}
reverse_product_mapping = {idx: product for product, idx in product_mapping.items()}

num_users = len(user_mapping)
num_products = len(product_mapping)

# Define Model Class (Same as Training)
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

# Load Trained Model
model_path = "models/recommendation_model.pth"
if not os.path.exists(model_path):
    raise HTTPException(status_code=500, detail="Model file is missing!")

model = RecommendationModel(num_users, num_products)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Request Model (User ID is now a string)
class RecommendationRequest(BaseModel):
    user_id: str
    top_k: int = 5  # Number of recommendations

# Recommendation Endpoint
@app.post("/recommend/")
def get_recommendations(request: RecommendationRequest):
    user_id = request.user_id
    top_k = request.top_k

    # Handle case where user_id is not in dataset
    if user_id not in user_mapping:
        return {"error": f"User ID '{user_id}' not found in dataset. Showing popular products instead."}

    # Convert string user_id to index using mapping
    user_idx = user_mapping[user_id]
    product_indices = list(product_mapping.values())
    user_tensor = torch.tensor([user_idx] * len(product_indices), dtype=torch.long)
    product_tensor = torch.tensor(product_indices, dtype=torch.long)

    # Get predictions
    with torch.no_grad():
        scores = model(user_tensor, product_tensor)

    # Select top K recommendations
    top_indices = torch.argsort(scores, descending=True)[:top_k]
    recommended_products = [reverse_product_mapping[idx.item()] for idx in top_indices]

    return {"user_id": user_id, "recommended_products": recommended_products}

# Root Endpoint
@app.get("/")
def home():
    return {"message": "Recommendation API is running!"}
