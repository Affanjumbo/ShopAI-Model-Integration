import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.data_loader import get_user_spending

def allocate_budget(user_id, budget):
    spending_pattern = get_user_spending(user_id)
    
    total_spent = sum(spending_pattern.values())
    if total_spent == 0:
        return {"message": "No purchase history found. Cannot allocate budget."}
    
    budget_allocation = {category: (amount / total_spent) * budget for category, amount in spending_pattern.items()}
    
    return budget_allocation
