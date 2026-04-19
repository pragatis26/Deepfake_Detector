"""
train.py — Fine-tune EfficientNet-B4 for DeepFake Detection
============================================================
Dataset structure expected:
    data/
        train/
            real/   (real face images)
            fake/   (deepfake face images)
        val/
            real/
            fake/

Recommended dataset: FaceForensics++ or Celeb-DF
Download: https://github.com/ondyari/FaceForensics

Usage:
    python train.py --data_dir data/ --epochs 10 --batch_size 32
"""

import os
import argparse
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from models.detector import EfficientNetDetector


# ── Transforms ────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_datasets(data_dir: str, batch_size: int):
    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=TRAIN_TRANSFORM)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"[Data] Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"[Data] Classes: {train_ds.class_to_idx}")
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = (outputs.detach().cpu() > 0.5).int().squeeze().tolist()
        all_preds.extend(preds if isinstance(preds, list) else [preds])
        all_labels.extend(labels.cpu().int().squeeze().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_scores = [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.float().unsqueeze(1).to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * imgs.size(0)

        scores = outputs.cpu().squeeze().tolist()
        preds  = [1 if s > 0.5 else 0 for s in (scores if isinstance(scores, list) else [scores])]
        all_scores.extend(scores if isinstance(scores, list) else [scores])
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().int().squeeze().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except Exception:
        auc = 0.0

    return epoch_loss, epoch_acc, auc, all_preds, all_labels


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")

    train_loader, val_loader = load_datasets(args.data_dir, args.batch_size)

    model = EfficientNetDetector().to(device)

    # Freeze base layers for first few epochs (feature extraction phase)
    for param in model.base.features.parameters():
        param.requires_grad = False

    criterion = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    print(f"\n{'='*60}")
    print(f"  Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):

        # Unfreeze all layers after epoch 3
        if epoch == 4:
            print("[Train] Unfreezing all layers for fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - 3)

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
              f"Time: {elapsed:.1f}s")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, 'models/efficientnet_deepfake.pth')
            print(f"  ✓ Best model saved (AUC: {val_auc:.4f})")

    # Final report
    model.load_state_dict(best_weights)
    _, _, _, preds, labels = evaluate(model, val_loader, criterion, device)
    print(f"\n{'='*60}")
    print("  Final Classification Report")
    print('='*60)
    print(classification_report(labels, preds, target_names=['Real', 'Fake']))
    print(f"  Best Val AUC: {best_val_auc:.4f}")
    print(f"  Model saved to: models/efficientnet_deepfake.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeepFake Detector')
    parser.add_argument('--data_dir',   type=str, default='data/',  help='Path to dataset')
    parser.add_argument('--epochs',     type=int, default=10,       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,       help='Batch size')
    parser.add_argument('--lr',         type=float, default=1e-4,   help='Learning rate')
    args = parser.parse_args()
    train(args)
