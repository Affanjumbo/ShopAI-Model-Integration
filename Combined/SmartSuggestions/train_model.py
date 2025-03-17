import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from SmartSuggestions.utils import load_dataset, build_stock_mappings, save_mappings, save_model
from SmartSuggestions.config import MODEL_PATH, EMBEDDING_DIM, BATCH_SIZE, EPOCHS, LEARNING_RATE, NEGATIVE_SAMPLES

class ProductDataset(Dataset):
    def __init__(self, transactions, stock_map):
        self.pairs = []
        self.stock_map = stock_map
        for transaction in transactions:
            encoded_transaction = [stock_map[stock] for stock in transaction if stock in stock_map]
            for i, stock in enumerate(encoded_transaction):
                for j in range(i + 1, len(encoded_transaction)):
                    self.pairs.append((stock, encoded_transaction[j]))
                    self.pairs.append((encoded_transaction[j], stock))

        self.negative_samples = list(stock_map.values())

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pos_pair = self.pairs[idx]
        neg_sample = random.choice(self.negative_samples)
        while neg_sample in pos_pair:
            neg_sample = random.choice(self.negative_samples)
        return torch.tensor(pos_pair[0]), torch.tensor(pos_pair[1]), torch.tensor(neg_sample)

class RecommendationModel(nn.Module):
    def __init__(self, num_products, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_products, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, num_products)

    def forward(self, stock_pos, stock_neg):
        pos_embedding = self.embeddings(stock_pos)
        neg_embedding = self.embeddings(stock_neg)
        return torch.sum(pos_embedding * neg_embedding, dim=1)

def train():
    df = load_dataset()
    stock_map, inverse_stock_map = build_stock_mappings(df)
    save_mappings(stock_map, inverse_stock_map)

    transactions = df.groupby("InvoiceNo")["StockCode"].apply(list).tolist()
    dataset = ProductDataset(transactions, stock_map)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = RecommendationModel(len(stock_map), EMBEDDING_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(EPOCHS):
        for stock_pos, stock_neg, neg_sample in dataloader:
            optimizer.zero_grad()
            pos_output = model(stock_pos, stock_neg)
            neg_output = model(stock_pos, neg_sample)
            loss = criterion(pos_output, torch.ones_like(pos_output)) + criterion(neg_output, torch.zeros_like(neg_output))
            loss.backward()
            optimizer.step()

    save_model(model, MODEL_PATH)

if __name__ == "__main__":
    train()
