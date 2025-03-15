import pandas as pd
import matplotlib.pyplot as plt

# Load the correct dataset
data = pd.read_csv("ExpenseManagerAI/datasets/user_budget_predictions.csv")

# Check if 'Predicted_Budget' exists
print("Columns in dataset:", data.columns)

# Scatter plot to visualize the relationship
plt.scatter(data["Avg_Monthly_Spending"], data["Predicted_Budget"], alpha=0.6, color='blue')
plt.xlabel("Avg Monthly Spending")
plt.ylabel("Predicted Budget")
plt.title("Relationship Between Monthly Spending and Predicted Budget")
plt.grid(True)
plt.show()
