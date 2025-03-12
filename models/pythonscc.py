import torch

# Load the saved model
model_path = "F:/ShopAI-AI-Models/models/recommendation_model.pth"
model_data = torch.load(model_path, map_location=torch.device('cpu'))

# Print the structure of the saved data
print("Keys in the saved model:", model_data.keys())

# If model_data is an OrderedDict, print the first few keys to inspect the structure
if isinstance(model_data, dict):
    for key in model_data.keys():
        print(f"\nKey: {key}\nValue Sample: {model_data[key]}")
