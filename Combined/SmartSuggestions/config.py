import os

# Dataset path
DATASET_PATH = "Combined/SmartSuggestions/datasets/cleaned_dataset.csv"

# Model and mappings
MODEL_PATH = "Combined/SmartSuggestions/models/smart_suggestions_model.pth"
STOCK_MAP_PATH = "Combined/SmartSuggestions/models/stock_map.pickle"
INVERSE_STOCK_MAP_PATH = "Combined/SmartSuggestions/models/inverse_stock_map.pickle"

# Training Hyperparameters
EMBEDDING_DIM = 128
BATCH_SIZE = 512
EPOCHS = 10
LEARNING_RATE = 0.001
NEGATIVE_SAMPLES = 5
