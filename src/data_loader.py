import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import BATCH_SIZE, IMG_SIZE, TEST_DIR, TRAIN_DIR


def validate_data_directories():
    for split_name, split_dir in [("training", TRAIN_DIR), ("testing", TEST_DIR)]:
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Expected {split_name} directory was not found: {split_dir}\n"
                "Run prepare_data.py before training or testing."
            )


def get_data_loaders():
    validate_data_directories()

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader, train_dataset.classes
