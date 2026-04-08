from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "bo-", label="Training Loss")
    plt.plot(epochs, val_losses, "ro-", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, "bo-", label="Training Acc")
    plt.plot(epochs, val_accs, "ro-", label="Validation Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_roc_curve(all_labels, all_probs, class_names):
    n_classes = len(class_names)
    y_test = label_binarize(all_labels, classes=range(n_classes))

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = cycle(
        ["blue", "red", "green", "orange", "purple", "brown", "teal", "pink", "gray"]
    )
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve of {class_names[i]} (area = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def evaluate_model(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)

    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Figure: Confusion Matrix")
    plt.tight_layout()
    plt.show()

    plot_roc_curve(all_labels, all_probs, class_names)


def apply_gradcam(model, input_tensor, target_class=None):
    model.eval()

    features = model.features(input_tensor.unsqueeze(0))
    if getattr(model, "use_attention", False):
        features = model.attention(features)
    features.retain_grad()

    pooled = model.avgpool(features)
    output = model.classifier(pooled)
    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, target_class].backward()

    grads = features.grad.data
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    for i in range(features.shape[1]):
        features.data[0, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(features, dim=1).squeeze()
    heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
    max_value = np.max(heatmap)
    if max_value > 0:
        heatmap /= max_value

    return heatmap
