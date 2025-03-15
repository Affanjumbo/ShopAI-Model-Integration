from fastapi import FastAPI
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from Combined.ExpenseManagerAI.main import app as expense_manager_app
from Combined.RecommendationAI.models.recommendation_api import app as recommendation_app

app = FastAPI()


app.mount("/expense", expense_manager_app)


app.mount("/recommendation", recommendation_app)

@app.get("/")
def root():
    return {"message": "Combined APIs are running!"}
