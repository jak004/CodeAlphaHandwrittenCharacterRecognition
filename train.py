"""
Train a CNN on MNIST (digits) or EMNIST-Balanced (digits + letters).

Usage:
    python train.py --dataset emnist --epochs 15
    python train.py --dataset mnist  --epochs 10
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import CharCNN

# EMNIST Balanced: 47 classes = 10 digits + 26 uppercase + 11 distinct lowercase
EMNIST_CLASSES = (
    list("0123456789")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abdefghnqrt")   # lowercase chars visually distinct from uppercase
)

MNIST_CLASSES = list("0123456789")


def get_loaders(dataset_name: str, batch_size: int, data_dir: str = "./data"):
    mean, std = (0.5,), (0.5,)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dataset_name == "mnist":
        train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
        test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        class_names = MNIST_CLASSES

    elif dataset_name == "emnist":
        train_ds = datasets.EMNIST(data_dir, split="balanced", train=True,  download=True, transform=transform)
        test_ds  = datasets.EMNIST(data_dir, split="balanced", train=False, download=True, transform=transform)
        class_names = EMNIST_CLASSES

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader, class_names


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser(description="Train handwritten character recognition CNN")
    parser.add_argument("--dataset",    choices=["mnist", "emnist"], default="emnist")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--data-dir",   default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Dataset: {args.dataset}")

    train_loader, test_loader, class_names = get_loaders(args.dataset, args.batch_size, args.data_dir)
    num_classes = len(class_names)
    print(f"Classes: {num_classes}  ({', '.join(class_names[:10])}{'...' if num_classes > 10 else ''})")

    model = CharCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    save_path = f"model_{args.dataset}.pth"
    best_acc = 0.0

    print("\nEpoch | Train Loss | Train Acc | Val Loss | Val Acc")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        marker = " *" if vl_acc > best_acc else ""
        print(f"  {epoch:02d}  | {tr_loss:.4f}     | {tr_acc:.4f}    | {vl_loss:.4f}   | {vl_acc:.4f}{marker}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save({
                "model_state": model.state_dict(),
                "num_classes":  num_classes,
                "class_names":  class_names,
                "dataset":      args.dataset,
            }, save_path)

    print(f"\nBest validation accuracy : {best_acc*100:.2f}%")
    print(f"Model saved to           : {save_path}")


if __name__ == "__main__":
    main()
