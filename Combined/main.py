from fastapi import FastAPI
import sys
import os

# Ensure the project root is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import existing APIs
from Combined.ExpenseManagerAI.main import app as expense_manager_app
from Combined.RecommendationAI.models.recommendation_api import app as recommendation_app
from Combined.SmartSuggestions import app as smart_suggestions_app  # Import Smart Suggestions API

app = FastAPI()

# Mount existing APIs
app.mount("/expense", expense_manager_app)
app.mount("/recommendation", recommendation_app)
app.mount("/smart_suggestions", smart_suggestions_app)  # Mount Smart Suggestions API

@app.get("/")
def root():
    return {"message": "Combined APIs are running!"}
