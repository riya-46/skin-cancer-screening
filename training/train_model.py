import json
import os
import random
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


# =========================
# 1. Reproducibility
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 2. Dataset
# =========================
class SkinCancerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================
# 3. Data Helpers
# =========================
def discover_class_names(root_dirs):
    priority = ["benign", "malignant", "invalid", "unknown", "not_skin_lesion"]
    discovered = []

    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            continue

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir) and class_name not in discovered:
                discovered.append(class_name)

    ordered = [name for name in priority if name in discovered]
    extras = [name for name in discovered if name not in ordered]
    class_names = ordered + extras

    if "benign" not in class_names or "malignant" not in class_names:
        raise ValueError(
            "Dataset must contain at least 'benign' and 'malignant' folders."
        )

    return class_names


def load_image_paths_and_labels(root_dir, class_to_idx):
    image_paths = []
    labels = []

    for class_name, label in class_to_idx.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file_name in sorted(os.listdir(class_dir)):
            file_path = os.path.join(class_dir, file_name)
            if os.path.isfile(file_path):
                image_paths.append(file_path)
                labels.append(label)

    return image_paths, labels


def print_split_stats(split_name, labels, class_names):
    counts = Counter(labels)
    print(f"{split_name} size:", len(labels))
    for idx, class_name in enumerate(class_names):
        print(f"{split_name} {class_name} count:", counts.get(idx, 0))


def build_transforms(img_size):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.72, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.06, 0.06),
            scale=(0.92, 1.08),
            shear=8,
        ),
        transforms.ColorJitter(
            brightness=0.22,
            contrast=0.22,
            saturation=0.18,
            hue=0.03,
        ),
        transforms.RandomPerspective(distortion_scale=0.18, p=0.25),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.20,
            scale=(0.02, 0.10),
            ratio=(0.3, 3.0),
            value="random",
        ),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


def build_weighted_sampler(labels):
    label_counts = Counter(labels)
    sample_weights = [1.0 / label_counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_class_weights(labels, num_classes, device):
    label_counts = Counter(labels)
    total_samples = len(labels)
    weights = []

    for class_idx in range(num_classes):
        count = label_counts.get(class_idx, 0)
        weight = total_samples / max(count * num_classes, 1)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32, device=device)


# =========================
# 4. Evaluation
# =========================
def evaluate_model(model, loader, criterion, device, malignant_idx):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_probs = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probs, dim=1)
            malignant_probs = probs[:, malignant_idx]

            total_loss += loss.item()
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(malignant_probs.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    acc = 100 * total_correct / max(total_samples, 1)
    binary_labels = (np.array(all_labels) == malignant_idx).astype(int)
    binary_preds = (np.array(all_preds) == malignant_idx).astype(int)
    malignant_precision = precision_score(
        binary_labels,
        binary_preds,
        zero_division=0,
    )
    malignant_recall = recall_score(
        binary_labels,
        binary_preds,
        zero_division=0,
    )
    malignant_f1 = f1_score(
        binary_labels,
        binary_preds,
        zero_division=0,
    )

    return {
        "loss": avg_loss,
        "acc": acc,
        "malignant_precision": malignant_precision,
        "malignant_recall": malignant_recall,
        "malignant_f1": malignant_f1,
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
        "confidences": np.array(all_confidences),
    }


def compute_binary_metrics(binary_labels, binary_preds):
    return {
        "precision": precision_score(binary_labels, binary_preds, zero_division=0),
        "recall": recall_score(binary_labels, binary_preds, zero_division=0),
        "f1": f1_score(binary_labels, binary_preds, zero_division=0),
    }


def find_best_malignant_threshold(all_labels, all_probs, malignant_idx):
    binary_labels = (np.array(all_labels) == malignant_idx).astype(int)
    candidate_thresholds = [0.35, 0.45, 0.55, 0.65, 0.75, 0.80]

    best_threshold = 0.75
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    best_score = -1.0

    for threshold in candidate_thresholds:
        binary_preds = (all_probs >= threshold).astype(int)
        metrics = compute_binary_metrics(binary_labels, binary_preds)

        # Favor precision improvement but keep recall clinically useful.
        score = (0.45 * metrics["precision"]) + (0.30 * metrics["recall"]) + (0.25 * metrics["f1"])
        if metrics["recall"] < 0.80:
            score -= 0.10

        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def find_best_confidence_threshold(metrics, malignant_idx):
    labels = metrics["labels"]
    preds = metrics["preds"]
    confidences = metrics["confidences"]
    candidate_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    best_threshold = 0.70
    best_score = -1.0

    for threshold in candidate_thresholds:
        keep_mask = confidences >= threshold
        coverage = keep_mask.mean()
        if coverage < 0.70:
            continue

        kept_labels = labels[keep_mask]
        kept_preds = preds[keep_mask]
        kept_binary_labels = (kept_labels == malignant_idx).astype(int)
        kept_binary_preds = (kept_preds == malignant_idx).astype(int)
        kept_metrics = compute_binary_metrics(kept_binary_labels, kept_binary_preds)
        kept_acc = (kept_labels == kept_preds).mean()

        score = (
            0.35 * kept_acc
            + 0.25 * kept_metrics["precision"]
            + 0.20 * kept_metrics["recall"]
            + 0.10 * kept_metrics["f1"]
            + 0.10 * coverage
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


def analyze_uncertainty(all_labels, all_probs, malignant_idx):
    thresholds = [0.35, 0.45, 0.55, 0.65, 0.75]

    print("\n=== Malignant Threshold Analysis ===")
    for threshold in thresholds:
        binary_labels = (np.array(all_labels) == malignant_idx).astype(int)
        binary_preds = (all_probs >= threshold).astype(int)
        positive_precision = precision_score(
            binary_labels,
            binary_preds,
            zero_division=0,
        )
        positive_recall = recall_score(
            binary_labels,
            binary_preds,
            zero_division=0,
        )
        positive_f1 = f1_score(
            binary_labels,
            binary_preds,
            zero_division=0,
        )

        print(
            f"Threshold {threshold:.2f} -> "
            f"Precision: {positive_precision:.4f}, "
            f"Recall: {positive_recall:.4f}, "
            f"F1: {positive_f1:.4f}"
        )


def analyze_confidence_rejection(metrics, class_names, malignant_idx):
    labels = metrics["labels"]
    preds = metrics["preds"]
    confidences = metrics["confidences"]

    rejection_thresholds = [0.50, 0.60, 0.70, 0.80]

    print("\n=== Confidence Rejection Analysis ===")
    for threshold in rejection_thresholds:
        keep_mask = confidences >= threshold
        kept = int(keep_mask.sum())
        total = len(confidences)

        if kept == 0:
            print(f"Confidence >= {threshold:.2f} -> no predictions kept")
            continue

        kept_labels = labels[keep_mask]
        kept_preds = preds[keep_mask]
        kept_acc = (kept_labels == kept_preds).mean() * 100.0
        kept_binary_labels = (kept_labels == malignant_idx).astype(int)
        kept_binary_preds = (kept_preds == malignant_idx).astype(int)
        kept_recall = recall_score(
            kept_binary_labels,
            kept_binary_preds,
            zero_division=0,
        )
        kept_f1 = f1_score(
            kept_binary_labels,
            kept_binary_preds,
            zero_division=0,
        )

        print(
            f"Confidence >= {threshold:.2f} -> "
            f"coverage: {kept}/{total} ({(kept / total) * 100:.2f}%), "
            f"acc: {kept_acc:.2f}%, "
            f"malignant recall: {kept_recall:.4f}, "
            f"malignant f1: {kept_f1:.4f}"
        )

    if any(name in class_names for name in ["invalid", "unknown", "not_skin_lesion"]):
        print(
            "\nInvalid-image class detected. The model can now learn to reject non-lesion uploads."
        )
    else:
        print(
            "\nNo invalid-image class found in dataset. "
            "Add a folder like 'invalid' in train/test for real-world rejection."
        )


# =========================
# 5. Main Training Pipeline
# =========================
def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # -------------------------
    # Paths
    # -------------------------
    data_dir = "data/skin_cancer"
    train_root = os.path.join(data_dir, "train")
    test_root = os.path.join(data_dir, "test")
    model_save_path = "models/best_model.pth"
    metadata_save_path = "models/best_model_meta.json"

    os.makedirs("models", exist_ok=True)

    class_names = discover_class_names([train_root, test_root])
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    malignant_idx = class_to_idx["malignant"]

    print("Classes:", class_names)
    print("Class mapping:", class_to_idx)

    # -------------------------
    # Hyperparameters
    # -------------------------
    img_size = 300
    batch_size = 16
    num_epochs = 25
    learning_rate = 3e-4
    weight_decay = 1e-4
    patience = 6

    # -------------------------
    # Load paths
    # -------------------------
    train_paths, train_labels = load_image_paths_and_labels(train_root, class_to_idx)
    test_paths, test_labels = load_image_paths_and_labels(test_root, class_to_idx)

    train_img_paths, val_img_paths, train_y, val_y = train_test_split(
        train_paths,
        train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels,
    )

    print_split_stats("Train", train_y, class_names)
    print_split_stats("Val", val_y, class_names)
    print_split_stats("Test", test_labels, class_names)

    # -------------------------
    # Transforms
    # -------------------------
    train_transform, eval_transform = build_transforms(img_size)

    # -------------------------
    # Datasets
    # -------------------------
    train_dataset = SkinCancerDataset(train_img_paths, train_y, transform=train_transform)
    val_dataset = SkinCancerDataset(val_img_paths, val_y, transform=eval_transform)
    test_dataset = SkinCancerDataset(test_paths, test_labels, transform=eval_transform)

    # -------------------------
    # DataLoaders
    # -------------------------
    train_sampler = build_weighted_sampler(train_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # -------------------------
    # Model
    # -------------------------
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(model.classifier.in_features, len(class_names)),
    )
    model = model.to(device)

    # -------------------------
    # Loss / Optimizer
    # -------------------------
    class_weights = build_class_weights(train_y, len(class_names), device)
    print("Class weights:", class_weights.detach().cpu().numpy().round(4).tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # -------------------------
    # Training Loop
    # -------------------------
    best_score = -1.0
    best_val_acc = 0.0
    best_val_recall = 0.0
    best_val_f1 = 0.0
    best_val_precision = 0.0
    best_malignant_threshold = 0.75
    best_confidence_threshold = 0.70
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = 100 * train_correct / max(train_total, 1)

        val_metrics = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            malignant_idx=malignant_idx,
        )
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]
        val_precision = val_metrics["malignant_precision"]
        val_recall = val_metrics["malignant_recall"]
        val_f1 = val_metrics["malignant_f1"]

        val_threshold, val_threshold_metrics = find_best_malignant_threshold(
            val_metrics["labels"],
            val_metrics["probs"],
            malignant_idx,
        )
        val_confidence_threshold = find_best_confidence_threshold(val_metrics, malignant_idx)

        score = (
            0.20 * (val_acc / 100.0)
            + 0.25 * val_precision
            + 0.30 * val_recall
            + 0.25 * val_f1
        )
        scheduler.step(score)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val Malignant Precision: {val_precision:.4f} | "
            f"Val Malignant Recall: {val_recall:.4f} | "
            f"Val Malignant F1: {val_f1:.4f} | "
            f"Best Thresh: {val_threshold:.2f}"
        )

        if score > best_score:
            best_score = score
            best_val_acc = val_acc
            best_val_precision = val_precision
            best_val_recall = val_recall
            best_val_f1 = val_f1
            best_malignant_threshold = val_threshold
            best_confidence_threshold = val_confidence_threshold
            no_improve_epochs = 0

            torch.save(model.state_dict(), model_save_path)
            with open(metadata_save_path, "w", encoding="utf-8") as meta_file:
                json.dump(
                    {
                        "class_names": class_names,
                        "class_to_idx": class_to_idx,
                        "img_size": img_size,
                        "recommended_malignant_threshold": best_malignant_threshold,
                        "recommended_confidence_threshold": best_confidence_threshold,
                        "validation_malignant_precision": best_val_precision,
                        "validation_malignant_recall": best_val_recall,
                        "validation_malignant_f1": best_val_f1,
                    },
                    meta_file,
                    indent=2,
                )
            print("Best model saved!")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")

        if no_improve_epochs >= patience:
            print("\nEarly stopping triggered.")
            break

    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation Malignant Precision: {best_val_precision:.4f}")
    print(f"Best Validation Malignant Recall: {best_val_recall:.4f}")
    print(f"Best Validation Malignant F1: {best_val_f1:.4f}")
    print(f"Recommended Malignant Threshold: {best_malignant_threshold:.2f}")
    print(f"Recommended Confidence Threshold: {best_confidence_threshold:.2f}")

    # -------------------------
    # Final Test Evaluation
    # -------------------------
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    test_metrics = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        malignant_idx=malignant_idx,
    )

    all_preds = test_metrics["preds"]
    all_labels = test_metrics["labels"]
    all_probs = test_metrics["probs"]

    print(f"\nFinal Test Accuracy: {test_metrics['acc']:.2f}%")
    print(f"Final Test Malignant Precision: {test_metrics['malignant_precision']:.4f}")
    print(f"Final Test Malignant Recall: {test_metrics['malignant_recall']:.4f}")
    print(f"Final Test Malignant F1: {test_metrics['malignant_f1']:.4f}")

    test_threshold, test_threshold_metrics = find_best_malignant_threshold(
        all_labels,
        all_probs,
        malignant_idx,
    )
    print(
        "Best Test Threshold Suggestion: "
        f"{test_threshold:.2f} "
        f"(Precision: {test_threshold_metrics['precision']:.4f}, "
        f"Recall: {test_threshold_metrics['recall']:.4f}, "
        f"F1: {test_threshold_metrics['f1']:.4f})"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    analyze_uncertainty(all_labels, all_probs, malignant_idx)
    analyze_confidence_rejection(test_metrics, class_names, malignant_idx)


if __name__ == "__main__":
    main()
