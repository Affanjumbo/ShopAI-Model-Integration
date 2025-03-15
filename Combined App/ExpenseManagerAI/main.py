import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils.expense_helper import recommend_products_with_pattern
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Expense Manager API is running!"}

class BudgetRequest(BaseModel):
    user_id: str
    budget: float

@app.post("/recommend")
def recommend_items(request: BudgetRequest):
    recommendations = recommend_products_with_pattern(request.user_id, request.budget)
    return recommendations
