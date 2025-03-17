import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from SmartSuggestions.train_model import RecommendationModel
from SmartSuggestions.utils import load_model, load_mappings
from SmartSuggestions.config import MODEL_PATH

class RecommendationPredictor:
    def __init__(self):
        self.stock_map, self.inverse_stock_map = load_mappings()
        self.model = RecommendationModel(len(self.stock_map), 128)
        self.model = load_model(self.model, MODEL_PATH)

    def recommend(self, stock_code, top_k=5):
        if stock_code not in self.stock_map:
            return []

        stock_idx = torch.tensor([self.stock_map[stock_code]])
        all_indices = torch.tensor(list(self.stock_map.values()))

        with torch.no_grad():
            stock_embedding = self.model.embeddings(stock_idx)
            all_embeddings = self.model.embeddings(all_indices)
            scores = F.cosine_similarity(stock_embedding, all_embeddings)

        scores = scores.numpy()
        sorted_indices = scores.argsort()[::-1]  # Sort in descending order

        recommendations = [
            (self.inverse_stock_map[idx], float(scores[idx]))  
            for idx in sorted_indices if idx != self.stock_map[stock_code]
        ][:top_k]  


        return recommendations

if __name__ == "__main__":
    predictor = RecommendationPredictor()
    stock_code = input("Enter StockCode: ").strip()
    recommendations = predictor.recommend(stock_code, top_k=5)

    if recommendations:
        print("\nTop 5 Recommended Products:")
        for stock, score in recommendations:
            print(f"StockCode: {stock}, Confidence: {score:.4f}")
    else:
        print("No recommendations found.")
