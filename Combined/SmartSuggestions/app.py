from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from SmartSuggestions.predict import RecommendationPredictor

app = FastAPI()
predictor = RecommendationPredictor()

class RecommendationRequest(BaseModel):
    stock_code: str
    top_k: int = 5  # Default to 5 recommendations

@app.post("/recommend/")
def get_recommendations(request: RecommendationRequest):
    recommendations = predictor.recommend(request.stock_code, request.top_k)
    return {"stock_code": request.stock_code, "recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
