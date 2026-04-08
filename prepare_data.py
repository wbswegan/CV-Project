import argparse
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"
METADATA_CSV = ARCHIVE_DIR / "HAM10000_metadata.csv"
PART1_DIR = ARCHIVE_DIR / "HAM10000_images_part_1"
PART2_DIR = ARCHIVE_DIR / "HAM10000_images_part_2"
TRAIN_ROOT = ARCHIVE_DIR / "Training"
TEST_ROOT = ARCHIVE_DIR / "Testing"


def validate_inputs():
    missing_paths = [
        path
        for path in [METADATA_CSV, PART1_DIR, PART2_DIR]
        if not path.exists()
    ]
    if missing_paths:
        missing_text = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Required dataset files were not found:\n{missing_text}")


def resolve_image_path(image_id):
    image_name = f"{image_id}.jpg"
    for directory in (PART1_DIR, PART2_DIR):
        candidate = directory / image_name
        if candidate.exists():
            return candidate
    return None


def build_grouped_split(metadata, random_state):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(
        splitter.split(metadata, y=metadata["dx"], groups=metadata["lesion_id"])
    )
    return (
        metadata.iloc[train_idx].reset_index(drop=True),
        metadata.iloc[test_idx].reset_index(drop=True),
    )


def reset_directory(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_subset(subset, destination_root):
    copied = 0
    missing_images = []

    for _, row in subset.iterrows():
        image_id = row["image_id"]
        label = row["dx"]
        source = resolve_image_path(image_id)
        target_dir = destination_root / label
        target_dir.mkdir(parents=True, exist_ok=True)

        if source is None:
            missing_images.append(image_id)
            continue

        shutil.copy2(source, target_dir / source.name)
        copied += 1

    return copied, missing_images


def print_split_summary(train_df, test_df):
    print("Dataset split summary")
    print(f"  Training images: {len(train_df)}")
    print(f"  Testing images:  {len(test_df)}")
    print("  Training class counts:", train_df["dx"].value_counts().sort_index().to_dict())
    print("  Testing class counts: ", test_df["dx"].value_counts().sort_index().to_dict())
    print(f"  Training lesions: {train_df['lesion_id'].nunique()}")
    print(f"  Testing lesions:  {test_df['lesion_id'].nunique()}")
    print(
        "  Lesion overlap:",
        len(set(train_df["lesion_id"]) & set(test_df["lesion_id"])),
    )


def organize_data(random_state=42):
    validate_inputs()
    metadata = pd.read_csv(METADATA_CSV)
    train_df, test_df = build_grouped_split(metadata, random_state=random_state)

    reset_directory(TRAIN_ROOT)
    reset_directory(TEST_ROOT)

    train_copied, train_missing = copy_subset(train_df, TRAIN_ROOT)
    test_copied, test_missing = copy_subset(test_df, TEST_ROOT)

    print_split_summary(train_df, test_df)
    print(f"Copied training images: {train_copied}")
    print(f"Copied testing images:  {test_copied}")

    missing_images = train_missing + test_missing
    if missing_images:
        print(f"Missing images: {len(missing_images)}")
        print("First few missing image ids:", missing_images[:10])
    else:
        print("All metadata rows were matched to image files.")

    print(f"\nTraining folder: {TRAIN_ROOT}")
    print(f"Testing folder:  {TEST_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare HAM10000 train/test folders.")
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used by the grouped splitter.",
    )
    args = parser.parse_args()
    organize_data(random_state=args.random_state)
