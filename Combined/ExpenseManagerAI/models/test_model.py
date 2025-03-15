import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# Load processed features
data = pd.read_csv("ExpenseManagerAI/datasets/user_features.csv")


# Features and target
features = ["Avg_Monthly_Spending", "Discounted_Purchases"]
target = "Avg_Monthly_Spending"

# Normalize features (use the same scaler from training)
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Prepare test data
X_test = torch.tensor(data[features].values, dtype=torch.float32)

# Load trained model
class BudgetPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super(BudgetPredictor, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


model = BudgetPredictor(input_dim=len(features))
model.load_state_dict(torch.load("ExpenseManagerAI/models/budget_predictor.pth"))
model.eval()

# Generate predictions
with torch.no_grad():
    predictions = model(X_test).squeeze().numpy()
predictions = scaler.inverse_transform(np.column_stack([predictions, predictions]))[:, 0]




# Save predictions
data["Predicted_Budget"] = predictions
data.to_csv("ExpenseManagerAI/datasets/user_budget_predictions.csv", index=False)

print("Predictions saved successfully.")
