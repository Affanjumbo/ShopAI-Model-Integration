import pandas as pd
import pickle
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from SmartSuggestions.config import DATASET_PATH, STOCK_MAP_PATH, INVERSE_STOCK_MAP_PATH

def load_dataset():
    df = pd.read_csv(DATASET_PATH, usecols=["InvoiceNo", "StockCode"])
    df.dropna(inplace=True)
    df["StockCode"] = df["StockCode"].astype(str)
    return df

def build_stock_mappings(df):
    unique_products = df["StockCode"].unique().tolist()
    stock_map = {stock: idx for idx, stock in enumerate(unique_products)}
    inverse_stock_map = {idx: stock for stock, idx in stock_map.items()}
    return stock_map, inverse_stock_map

def save_mappings(stock_map, inverse_stock_map):
    with open(STOCK_MAP_PATH, "wb") as f:
        pickle.dump(stock_map, f)
    with open(INVERSE_STOCK_MAP_PATH, "wb") as f:
        pickle.dump(inverse_stock_map, f)

def load_mappings():
    with open(STOCK_MAP_PATH, "rb") as f:
        stock_map = pickle.load(f)
    with open(INVERSE_STOCK_MAP_PATH, "rb") as f:
        inverse_stock_map = pickle.load(f)
    return stock_map, inverse_stock_map

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
