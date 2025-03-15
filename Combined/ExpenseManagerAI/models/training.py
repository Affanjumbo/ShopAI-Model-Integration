import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load processed features
data = pd.read_csv("ExpenseManagerAI/datasets/user_features.csv")

# Features and target
features = ["Avg_Monthly_Spending", "Discounted_Purchases"]
target = "Avg_Monthly_Spending"  # Predicting the spending itself

# Normalize features
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Prepare training data
X = data[features].values
y = data[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define a simple Feedforward Neural Network
class BudgetPredictor(nn.Module):
    def __init__(self, input_dim):
        super(BudgetPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Initialize model, loss, and optimizer
model = BudgetPredictor(input_dim=len(features))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "ExpenseManagerAI/models/budget_predictor.pth")
