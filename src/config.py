import os

import torch

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Data and model paths
TRAIN_DIR = os.path.join(ARCHIVE_DIR, "Training")
TEST_DIR = os.path.join(ARCHIVE_DIR, "Testing")
MODEL_PATH = os.path.join(MODEL_DIR, "skin_cancer_model.pth")
HISTORY_DB_PATH = os.path.join(DATA_DIR, "history.db")
HISTORY_IMAGE_DIR = os.path.join(DATA_DIR, "history_images")
HISTORY_PAGE_SIZE = 20

# Frontend assets served by FastAPI
FRONTEND_INDEX = os.path.join(BASE_DIR, "index.html")
FRONTEND_SCRIPT = os.path.join(BASE_DIR, "script.js")
FRONTEND_STYLE = os.path.join(BASE_DIR, "style.css")

# Local demo server defaults
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Training hyperparameters
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-6
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 30

# Hardware device
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
