import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Recommendation Model
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

# Load Dataset
def load_data():
    try:
        df = pd.read_csv("F:/ShopAI-AI-Models/datasets/cleaned_ecommerce.csv")
        print("✅ Data loaded successfully.")

        # Rename Columns to Standardized Names
        df = df.rename(columns={"User_ID": "user_id", "Product_ID": "product_id"})

        # Generate Fake Ratings (1 to 5 stars) Since There’s No `rating` Column
        if "rating" not in df.columns:
            df["rating"] = np.random.randint(1, 6, size=len(df))

        df = df[['user_id', 'product_id', 'rating']].dropna()
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

# Train and Save Model
def train_and_save_model():
    df = load_data()
    if df is None:
        return
    
    user_mapping = {user: idx for idx, user in enumerate(df["user_id"].unique())}
    product_mapping = {product: idx for idx, product in enumerate(df["product_id"].unique())}
    
    df["user_id"] = df["user_id"].map(user_mapping)
    df["product_id"] = df["product_id"].map(product_mapping)

    num_users = len(user_mapping)
    num_products = len(product_mapping)

    model = RecommendationModel(num_users, num_products)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Convert data to PyTorch tensors
    user_tensor = torch.tensor(df["user_id"].values, dtype=torch.long)
    product_tensor = torch.tensor(df["product_id"].values, dtype=torch.long)
    rating_tensor = torch.tensor(df["rating"].values, dtype=torch.float32)

    # Training Loop
    epochs = 20
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(user_tensor, product_tensor)
        loss = criterion(predictions, rating_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save Model
    torch.save(model.state_dict(), "F:/ShopAI-AI-Models/models/recommendation_model.pth")
    print("✅ Model saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
