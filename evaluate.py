"""
evaluate.py — Evaluate the trained model and generate a full metrics report
===========================================================================
Usage:
    python evaluate.py --data_dir data/val --output_dir results/
"""

import os
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

from models.detector import EfficientNetDetector

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def evaluate(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = EfficientNetDetector().to(device)
    model.load_state_dict(torch.load('models/efficientnet_deepfake.pth', map_location=device))
    model.eval()
    print(f"Model loaded. Evaluating on: {args.data_dir}")

    # Dataset
    dataset = datasets.ImageFolder(args.data_dir, transform=VAL_TRANSFORM)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"Classes: {dataset.class_to_idx}  |  Total: {len(dataset)} images")

    all_scores, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            scores = model(imgs).squeeze().cpu().tolist()
            if not isinstance(scores, list):
                scores = [scores]
            preds = [1 if s > 0.5 else 0 for s in scores]
            all_scores.extend(scores)
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # ── Metrics ──────────────────────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_scores)
    cm  = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=list(dataset.class_to_idx.keys()))

    print(f"\nAccuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve — DeepFake Detector')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'roc_curve.png'), dpi=150)
    plt.close()

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Real', 'Fake']); ax.set_yticklabels(['Real', 'Fake'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Save JSON summary
    summary = {
        'accuracy': round(acc, 4),
        'roc_auc':  round(auc, 4),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_labels)
    }
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='data/val')
    parser.add_argument('--output_dir', default='results/')
    evaluate(parser.parse_args())
