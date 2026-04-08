import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import DEVICE, EPOCHS, LEARNING_RATE, MIN_LR, MODEL_PATH, MODEL_DIR, TEST_DIR, TRAIN_DIR
from src.data_loader import get_data_loaders
from src.model import get_model
from src.utils import evaluate_model


def validate_runtime_paths(require_model=False):
    for split_name, split_dir in [("training", TRAIN_DIR), ("testing", TEST_DIR)]:
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected {split_name} directory was not found: {split_dir}\n"
                "Run prepare_data.py first or verify the archive paths."
            )

    os.makedirs(MODEL_DIR, exist_ok=True)

    if require_model and not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights were not found: {MODEL_PATH}\n"
            "Train the model first or place the weight file in the models directory."
        )


def train():
    validate_runtime_paths()
    train_loader, _, classes = get_data_loaders()
    model = get_model(len(classes), pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=MIN_LR
    )

    print(f"Starting training on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def test():
    validate_runtime_paths(require_model=True)
    _, test_loader, classes = get_data_loaders()
    model = get_model(len(classes), pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    evaluate_model(model, test_loader, DEVICE, classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skin lesion classification workflow")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"])
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        test()
