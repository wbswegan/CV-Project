import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    average_precision_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEVICE, MODEL_PATH
from src.data_loader import get_data_loaders
from src.labels import CLASS_NAMES, DISEASE_INFO
from src.model import get_model


ARCHIVE_DIR = PROJECT_ROOT / "archive"
METADATA_CSV = ARCHIVE_DIR / "HAM10000_metadata.csv"
TRAIN_DIR = ARCHIVE_DIR / "Training"
TEST_DIR = ARCHIVE_DIR / "Testing"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "current_eda"


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_figure(name):
    plt.savefig(OUTPUT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def collect_split_records(split_name, split_dir):
    rows = []
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.glob("*.jpg")):
            rows.append(
                {
                    "image_id": image_path.stem,
                    "dx": class_name,
                    "split": split_name,
                    "path": str(image_path),
                }
            )
    return pd.DataFrame(rows)


def load_dataset_frame():
    metadata = pd.read_csv(METADATA_CSV)
    train_df = collect_split_records("Training", TRAIN_DIR)
    test_df = collect_split_records("Testing", TEST_DIR)
    split_df = pd.concat([train_df, test_df], ignore_index=True)

    dataset_df = split_df.merge(metadata, on="image_id", how="left", suffixes=("", "_meta"))

    if dataset_df["dx_meta"].isna().any():
        missing = dataset_df.loc[dataset_df["dx_meta"].isna(), "image_id"].tolist()[:10]
        raise ValueError(f"Metadata rows were missing for images: {missing}")

    mismatched = dataset_df.loc[dataset_df["dx"] != dataset_df["dx_meta"], ["image_id", "dx", "dx_meta"]]
    if not mismatched.empty:
        sample = mismatched.head(10).to_dict("records")
        raise ValueError(f"Folder labels and metadata labels differ: {sample}")

    dataset_df = dataset_df.drop(columns=["dx_meta"])
    dataset_df["sex"] = dataset_df["sex"].fillna("unknown")
    dataset_df["localization"] = dataset_df["localization"].fillna("unknown")
    dataset_df["dx_type"] = dataset_df["dx_type"].fillna("unknown")
    dataset_df["age"] = pd.to_numeric(dataset_df["age"], errors="coerce")

    return dataset_df


def build_summary(dataset_df):
    train_df = dataset_df[dataset_df["split"] == "Training"].copy()
    test_df = dataset_df[dataset_df["split"] == "Testing"].copy()

    train_images = set(train_df["image_id"])
    test_images = set(test_df["image_id"])
    train_lesions = set(train_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    size_counter = dataset_df["path"].map(read_image_size).value_counts().sort_values(ascending=False)

    return {
        "totals": {
            "training_images": int(len(train_df)),
            "testing_images": int(len(test_df)),
            "training_lesions": int(train_df["lesion_id"].nunique()),
            "testing_lesions": int(test_df["lesion_id"].nunique()),
            "image_overlap": int(len(train_images & test_images)),
            "lesion_overlap": int(len(train_lesions & test_lesions)),
        },
        "class_counts": {
            "training": train_df["dx"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict(),
            "testing": test_df["dx"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict(),
        },
        "metadata": {
            "sex": dataset_df["sex"].value_counts().to_dict(),
            "localization_top10": dataset_df["localization"].value_counts().head(10).to_dict(),
            "dx_type": dataset_df["dx_type"].value_counts().to_dict(),
            "age_non_null": int(dataset_df["age"].notna().sum()),
            "age_median": float(dataset_df["age"].median()),
        },
        "image_sizes": {f"{width}x{height}": int(count) for (width, height), count in size_counter.items()},
    }


def read_image_size(image_path):
    with Image.open(image_path) as image:
        return image.size


def plot_class_distribution(dataset_df):
    counts = (
        dataset_df.groupby(["dx", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(CLASS_NAMES)
        .reindex(columns=["Training", "Testing"])
    )
    x = np.arange(len(CLASS_NAMES))
    width = 0.36

    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, counts["Training"], width, label="Training", color="steelblue")
    plt.bar(x + width / 2, counts["Testing"], width, label="Testing", color="coral")
    plt.xticks(x, CLASS_NAMES, rotation=45, ha="right")
    plt.ylabel("Number of images")
    plt.xlabel("Class")
    plt.title("Class distribution (current grouped split)")
    plt.legend()
    plt.tight_layout()
    save_figure("class_distribution.png")


def plot_split_overview(dataset_df):
    split_order = ["Training", "Testing"]
    image_counts = dataset_df["split"].value_counts().reindex(split_order)
    lesion_counts = dataset_df.groupby("split")["lesion_id"].nunique().reindex(split_order)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].bar(image_counts.index, image_counts.values, color=["steelblue", "coral"])
    axes[0].set_title("Images per split")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")

    axes[1].bar(lesion_counts.index, lesion_counts.values, color=["steelblue", "coral"])
    axes[1].set_title("Unique lesions per split")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Count")

    for ax, values in zip(axes, [image_counts.values, lesion_counts.values]):
        for idx, value in enumerate(values):
            ax.text(idx, value + max(values) * 0.02, f"{int(value)}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Dataset split overview")
    fig.tight_layout()
    save_figure("split_overview.png")


def plot_split_integrity(dataset_df):
    train_df = dataset_df[dataset_df["split"] == "Training"]
    test_df = dataset_df[dataset_df["split"] == "Testing"]

    overlap_images = len(set(train_df["image_id"]) & set(test_df["image_id"]))
    overlap_lesions = len(set(train_df["lesion_id"]) & set(test_df["lesion_id"]))
    leakage_counts = pd.Series(
        {
            "Image overlap": overlap_images,
            "Lesion overlap": overlap_lesions,
        }
    )
    lesion_counts = pd.Series(
        {
            "Training lesions": train_df["lesion_id"].nunique(),
            "Testing lesions": test_df["lesion_id"].nunique(),
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].bar(leakage_counts.index, leakage_counts.values, color=["seagreen", "darkorange"])
    axes[0].set_title("Leakage check")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=12)
    leakage_pad = max(0.08, float(max(leakage_counts.values)) * 0.08)
    axes[0].set_ylim(0, max(1.0, float(max(leakage_counts.values)) + leakage_pad * 3))
    for idx, value in enumerate(leakage_counts.values):
        axes[0].text(idx, value + leakage_pad, f"{int(value)}", ha="center", va="bottom", fontsize=10)

    axes[1].bar(lesion_counts.index, lesion_counts.values, color=["steelblue", "coral"])
    axes[1].set_title("Unique lesion counts")
    axes[1].set_ylabel("Count")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=12)
    for idx, value in enumerate(lesion_counts.values):
        axes[1].text(idx, value + max(lesion_counts.values) * 0.02, f"{int(value)}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("Current split integrity")
    fig.subplots_adjust(top=0.84, bottom=0.18, wspace=0.28)
    save_figure("split_integrity.png")


def plot_age_distribution(dataset_df):
    age_df = dataset_df.dropna(subset=["age"])
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=age_df,
        x="age",
        hue="split",
        bins=20,
        multiple="layer",
        alpha=0.45,
        palette={"Training": "steelblue", "Testing": "coral"},
    )
    plt.title("Age distribution by split")
    plt.xlabel("Age")
    plt.ylabel("Number of images")
    plt.tight_layout()
    save_figure("age_distribution.png")


def plot_sex_distribution(dataset_df):
    sex_order = ["male", "female", "unknown"]
    counts = (
        dataset_df.assign(sex=dataset_df["sex"].str.lower())
        .groupby(["sex", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(sex_order, fill_value=0)
        .reindex(columns=["Training", "Testing"])
    )
    x = np.arange(len(sex_order))
    width = 0.36

    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width / 2, counts["Training"], width, label="Training", color="steelblue")
    plt.bar(x + width / 2, counts["Testing"], width, label="Testing", color="coral")
    plt.xticks(x, [label.title() for label in sex_order])
    plt.ylabel("Number of images")
    plt.xlabel("Sex")
    plt.title("Sex distribution by split")
    plt.legend()
    plt.tight_layout()
    save_figure("sex_distribution.png")


def plot_localization_distribution(dataset_df):
    top_localizations = dataset_df["localization"].value_counts().head(10).index.tolist()
    counts = (
        dataset_df[dataset_df["localization"].isin(top_localizations)]
        .groupby(["localization", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(top_localizations)
        .reindex(columns=["Training", "Testing"])
    )
    x = np.arange(len(top_localizations))
    width = 0.36

    plt.figure(figsize=(12, 5.2))
    plt.bar(x - width / 2, counts["Training"], width, label="Training", color="steelblue")
    plt.bar(x + width / 2, counts["Testing"], width, label="Testing", color="coral")
    plt.xticks(x, top_localizations, rotation=40, ha="right")
    plt.ylabel("Number of images")
    plt.xlabel("Body location")
    plt.title("Top 10 localization counts by split")
    plt.legend()
    plt.tight_layout()
    save_figure("localization_distribution.png")


def plot_dx_type_distribution(dataset_df):
    dx_order = dataset_df["dx_type"].value_counts().index.tolist()
    counts = (
        dataset_df.groupby(["dx_type", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(dx_order)
        .reindex(columns=["Training", "Testing"])
    )
    x = np.arange(len(dx_order))
    width = 0.36

    plt.figure(figsize=(9.5, 5))
    plt.bar(x - width / 2, counts["Training"], width, label="Training", color="steelblue")
    plt.bar(x + width / 2, counts["Testing"], width, label="Testing", color="coral")
    plt.xticks(x, dx_order, rotation=30, ha="right")
    plt.ylabel("Number of images")
    plt.xlabel("Diagnosis type")
    plt.title("Diagnosis source distribution by split")
    plt.legend()
    plt.tight_layout()
    save_figure("dx_type_distribution.png")


def plot_sample_gallery(dataset_df, samples_per_class=3):
    train_df = dataset_df[dataset_df["split"] == "Training"]
    fig, axes = plt.subplots(len(CLASS_NAMES), samples_per_class, figsize=(samples_per_class * 3.2, len(CLASS_NAMES) * 2.4))

    if len(CLASS_NAMES) == 1:
        axes = np.array([axes])

    for row_idx, class_name in enumerate(CLASS_NAMES):
        class_rows = train_df[train_df["dx"] == class_name].sort_values("image_id").head(samples_per_class)
        for col_idx in range(samples_per_class):
            ax = axes[row_idx, col_idx]
            ax.axis("off")
            if col_idx < len(class_rows):
                image_path = class_rows.iloc[col_idx]["path"]
                with Image.open(image_path) as image:
                    ax.imshow(image.convert("RGB"))
            if col_idx == 0:
                ax.text(
                    -0.18,
                    0.5,
                    class_name,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )

    for col_idx in range(samples_per_class):
        axes[0, col_idx].set_title(f"Sample {col_idx + 1}")

    fig.suptitle("Training set sample gallery by class", y=0.995)
    fig.tight_layout(rect=(0.04, 0, 1, 0.985))
    save_figure("sample_gallery.png")


def save_summary(summary):
    with open(OUTPUT_DIR / "eda_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def collect_model_outputs():
    if not MODEL_PATH or not Path(MODEL_PATH).exists():
        print("Model weights were not found, skipping evaluation figures.")
        return None

    _, test_loader, classes = get_data_loaders()
    model = get_model(len(classes), pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return {
        "labels": np.array(all_labels),
        "preds": np.array(all_preds),
        "probs": np.array(all_probs),
        "classes": classes,
    }


def save_classification_report(labels, preds, classes):
    report_dict = classification_report(labels, preds, target_names=classes, output_dict=True, zero_division=0)
    report_text = classification_report(labels, preds, target_names=classes, zero_division=0)

    with open(OUTPUT_DIR / "classification_report.json", "w", encoding="utf-8") as handle:
        json.dump(report_dict, handle, indent=2, ensure_ascii=False)

    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as handle:
        handle.write(report_text)


def plot_confusion_matrix(labels, preds, classes):
    cm = confusion_matrix(labels, preds, labels=range(len(classes)))

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix (counts)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    save_figure("confusion_matrix.png")

    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm /= row_sums

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix (normalized by true class)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    save_figure("confusion_matrix_normalized.png")


def plot_per_class_metrics(labels, preds, classes):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=range(len(classes)),
        average=None,
        zero_division=0,
    )

    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(11, 5.2))
    plt.bar(x - width, precision, width, label="Precision", color="steelblue")
    plt.bar(x, recall, width, label="Recall", color="seagreen")
    plt.bar(x + width, f1, width, label="F1", color="coral")
    plt.xticks(x, classes, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title("Per-class Precision / Recall / F1")
    plt.legend()
    plt.tight_layout()
    save_figure("per_class_metrics.png")

    plt.figure(figsize=(10.5, 4.6))
    plt.bar(classes, support, color="slateblue")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Number of test samples")
    plt.title("Per-class support in the current test set")
    plt.tight_layout()
    save_figure("test_support.png")


def plot_roc_curves(labels, probs, classes):
    y_true = label_binarize(labels, classes=range(len(classes)))
    plt.figure(figsize=(8, 6))

    for idx, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true[:, idx], probs[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC (one-vs-rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    save_figure("roc_curves.png")


def plot_pr_curves(labels, probs, classes):
    y_true = label_binarize(labels, classes=range(len(classes)))
    plt.figure(figsize=(8, 6))

    for idx, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true[:, idx], probs[:, idx])
        ap = average_precision_score(y_true[:, idx], probs[:, idx])
        plt.plot(recall, precision, lw=2, label=f"{class_name} (AP = {ap:.2f})")

    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall (one-vs-rest)")
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    save_figure("pr_curves.png")


def main():
    ensure_output_dir()
    sns.set_theme(style="whitegrid")

    dataset_df = load_dataset_frame()
    summary = build_summary(dataset_df)

    plot_class_distribution(dataset_df)
    plot_split_overview(dataset_df)
    plot_split_integrity(dataset_df)
    plot_age_distribution(dataset_df)
    plot_sex_distribution(dataset_df)
    plot_localization_distribution(dataset_df)
    plot_dx_type_distribution(dataset_df)
    plot_sample_gallery(dataset_df)
    save_summary(summary)

    model_outputs = collect_model_outputs()
    if model_outputs is not None:
        save_classification_report(model_outputs["labels"], model_outputs["preds"], model_outputs["classes"])
        plot_confusion_matrix(model_outputs["labels"], model_outputs["preds"], model_outputs["classes"])
        plot_per_class_metrics(model_outputs["labels"], model_outputs["preds"], model_outputs["classes"])
        plot_roc_curves(model_outputs["labels"], model_outputs["probs"], model_outputs["classes"])
        plot_pr_curves(model_outputs["labels"], model_outputs["probs"], model_outputs["classes"])

    print(f"Saved current EDA assets to: {OUTPUT_DIR}")
    print(json.dumps(summary["totals"], indent=2))


if __name__ == "__main__":
    main()
