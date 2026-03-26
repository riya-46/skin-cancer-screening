import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B3_Weights


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def load_image_paths_and_labels(root_dir):
    class_to_idx = {"benign": 0, "malignant": 1}
    image_paths = []
    labels = []

    for class_name, label in class_to_idx.items():
        class_dir = os.path.join(root_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if os.path.isfile(file_path):
                image_paths.append(file_path)
                labels.append(label)

    return image_paths, labels


def evaluate_model(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, predicted = torch.max(outputs, 1)

            total_loss += loss.item()
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = 100 * total_correct / total_samples
    malignant_precision = precision_score(all_labels, all_preds, pos_label=1)
    malignant_recall = recall_score(all_labels, all_preds, pos_label=1)
    malignant_f1 = f1_score(all_labels, all_preds, pos_label=1)

    return {
        "loss": avg_loss,
        "acc": acc,
        "malignant_precision": malignant_precision,
        "malignant_recall": malignant_recall,
        "malignant_f1": malignant_f1,
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
    }


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    data_dir = "data/skin_cancer"
    train_root = os.path.join(data_dir, "train")
    test_root = os.path.join(data_dir, "test")
    model_save_path = "models/best_model_efficientnet_b3.pth"

    os.makedirs("models", exist_ok=True)

    IMG_SIZE = 300
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    PATIENCE = 5

    train_paths, train_labels = load_image_paths_and_labels(train_root)
    test_paths, test_labels = load_image_paths_and_labels(test_root)

    train_img_paths, val_img_paths, train_y, val_y = train_test_split(
        train_paths,
        train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels,
    )

    print("Train size:", len(train_img_paths))
    print("Val size:", len(val_img_paths))
    print("Test size:", len(test_paths))

    print("Train benign count:", sum(1 for y in train_y if y == 0))
    print("Train malignant count:", sum(1 for y in train_y if y == 1))
    print("Val benign count:", sum(1 for y in val_y if y == 0))
    print("Val malignant count:", sum(1 for y in val_y if y == 1))
    print("Test benign count:", sum(1 for y in test_labels if y == 0))
    print("Test malignant count:", sum(1 for y in test_labels if y == 1))

    weights = EfficientNet_B3_Weights.DEFAULT

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = SkinCancerDataset(train_img_paths, train_y, transform=train_transform)
    val_dataset = SkinCancerDataset(val_img_paths, val_y, transform=eval_transform)
    test_dataset = SkinCancerDataset(test_paths, test_labels, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    class_names = ["benign", "malignant"]
    print("Classes:", class_names)

    model = models.efficientnet_b3(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_score = -1.0
    best_val_acc = 0.0
    best_val_recall = 0.0
    best_val_f1 = 0.0
    no_improve_epochs = 0

    for epoch in range(NUM_EPOCHS):
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

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]
        val_recall = val_metrics["malignant_recall"]
        val_f1 = val_metrics["malignant_f1"]

        score = (0.4 * (val_acc / 100.0)) + (0.3 * val_recall) + (0.3 * val_f1)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Val Malignant Recall: {val_recall:.4f} | "
            f"Val Malignant F1: {val_f1:.4f}"
        )

        if score > best_score:
            best_score = score
            best_val_acc = val_acc
            best_val_recall = val_recall
            best_val_f1 = val_f1
            no_improve_epochs = 0

            torch.save(model.state_dict(), model_save_path)
            print("Best model saved!")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epoch(s).")

        if no_improve_epochs >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation Malignant Recall: {best_val_recall:.4f}")
    print(f"Best Validation Malignant F1: {best_val_f1:.4f}")

    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    all_preds = test_metrics["preds"]
    all_labels = test_metrics["labels"]
    all_probs = test_metrics["probs"]

    print(f"\nFinal Test Accuracy: {test_metrics['acc']:.2f}%")
    print(f"Final Test Malignant Precision: {test_metrics['malignant_precision']:.4f}")
    print(f"Final Test Malignant Recall: {test_metrics['malignant_recall']:.4f}")
    print(f"Final Test Malignant F1: {test_metrics['malignant_f1']:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    print("\n=== Threshold Analysis ===")
    for t in thresholds:
        preds_t = (all_probs >= t).astype(int)
        precision_t = precision_score(all_labels, preds_t, pos_label=1)
        recall_t = recall_score(all_labels, preds_t, pos_label=1)
        f1_t = f1_score(all_labels, preds_t, pos_label=1)

        print(
            f"Threshold {t} -> "
            f"Precision: {precision_t:.4f}, "
            f"Recall: {recall_t:.4f}, "
            f"F1: {f1_t:.4f}"
        )


if __name__ == "__main__":
    main()