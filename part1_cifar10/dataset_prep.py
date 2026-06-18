"""
CIFAR-10 Dataset Preparation
=============================
Downloads CIFAR-10, filters to 5 classes (airplane, automobile, bird, cat, deer),
splits into train/val/test, and reports statistics.

Strategy:
  - Keep the official CIFAR-10 test set as our test set (untouched).
  - Split the official train set into train/val using stratified sampling (85/15).

Usage:
    python dataset_prep.py
"""

import os
import sys
import numpy as np
import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLASSES_TO_KEEP = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer'}
CLASSES_TO_DELETE = {5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'data', 'filtered')

VAL_SPLIT = 0.15  # 15% of official train set -> validation


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  CIFAR-10 DATASET PREPARATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Download CIFAR-10
    # ------------------------------------------------------------------
    print("\n[1/5] Downloading CIFAR-10 dataset ...")
    train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True,  download=True)
    test_dataset  = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

    all_classes = train_dataset.classes
    print(f"  Original 10 classes : {all_classes}")
    print(f"  Official train size : {len(train_dataset):,}")
    print(f"  Official test  size : {len(test_dataset):,}")

    # ------------------------------------------------------------------
    # 2. Report which classes are deleted
    # ------------------------------------------------------------------
    print("\n[2/5] Removing unwanted classes ...")
    print(f"  Keeping  : {list(CLASSES_TO_KEEP.values())}")
    print(f"  Deleting : {list(CLASSES_TO_DELETE.values())}")

    keep_indices = set(CLASSES_TO_KEEP.keys())

    # Filter official train
    train_data    = train_dataset.data                      # (50000, 32, 32, 3)
    train_targets = np.array(train_dataset.targets)         # (50000,)
    train_mask    = np.isin(train_targets, list(keep_indices))
    filt_train_data    = train_data[train_mask]
    filt_train_targets = train_targets[train_mask]

    # Filter official test
    test_data    = test_dataset.data
    test_targets = np.array(test_dataset.targets)
    test_mask    = np.isin(test_targets, list(keep_indices))
    filt_test_data    = test_data[test_mask]
    filt_test_targets = test_targets[test_mask]

    # ------------------------------------------------------------------
    # 3. Report images left per class
    # ------------------------------------------------------------------
    total_filtered = len(filt_train_data) + len(filt_test_data)
    print(f"\n[3/5] Images remaining after filtering: {total_filtered:,}")
    print(f"\n  {'Class':<15} {'Official Train':>15} {'Official Test':>15} {'Total':>10}")
    print(f"  {'-' * 57}")
    for idx, name in sorted(CLASSES_TO_KEEP.items()):
        tr_cnt = int(np.sum(filt_train_targets == idx))
        te_cnt = int(np.sum(filt_test_targets == idx))
        print(f"  {name:<15} {tr_cnt:>15,} {te_cnt:>15,} {tr_cnt + te_cnt:>10,}")
    print(f"  {'-' * 57}")
    print(f"  {'TOTAL':<15} {len(filt_train_data):>15,} {len(filt_test_data):>15,} {total_filtered:>10,}")

    # ------------------------------------------------------------------
    # 4. Split official train -> train / val  (stratified)
    # ------------------------------------------------------------------
    print(f"\n[4/5] Splitting official train set -> train / val  "
          f"({int((1 - VAL_SPLIT) * 100)}/{int(VAL_SPLIT * 100)}) ...")

    train_idx, val_idx = train_test_split(
        np.arange(len(filt_train_data)),
        test_size=VAL_SPLIT,
        random_state=42,
        stratify=filt_train_targets,
    )

    final_train_data    = filt_train_data[train_idx]
    final_train_targets = filt_train_targets[train_idx]
    final_val_data      = filt_train_data[val_idx]
    final_val_targets   = filt_train_targets[val_idx]
    final_test_data     = filt_test_data
    final_test_targets  = filt_test_targets

    # ------------------------------------------------------------------
    # 5. Report final split counts
    # ------------------------------------------------------------------
    total_tr = len(final_train_data)
    total_va = len(final_val_data)
    total_te = len(final_test_data)
    grand    = total_tr + total_va + total_te

    print(f"\n[5/5] Final dataset statistics")
    print(f"\n  {'Class':<15} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'-' * 50}")
    for idx, name in sorted(CLASSES_TO_KEEP.items()):
        tr = int(np.sum(final_train_targets == idx))
        va = int(np.sum(final_val_targets   == idx))
        te = int(np.sum(final_test_targets  == idx))
        print(f"  {name:<15} {tr:>8,} {va:>8,} {te:>8,} {tr + va + te:>8,}")
    print(f"  {'-' * 50}")
    print(f"  {'TOTAL':<15} {total_tr:>8,} {total_va:>8,} {total_te:>8,} {grand:>8,}")
    print(f"\n  Split %:  Train {total_tr / grand * 100:.1f}%  |  "
          f"Val {total_va / grand * 100:.1f}%  |  "
          f"Test {total_te / grand * 100:.1f}%")

    # ------------------------------------------------------------------
    # Save processed data
    # ------------------------------------------------------------------
    save_path = os.path.join(OUTPUT_DIR, 'cifar10_filtered.pt')
    torch.save({
        'train_data':    final_train_data,
        'train_targets': final_train_targets.tolist(),
        'val_data':      final_val_data,
        'val_targets':   final_val_targets.tolist(),
        'test_data':     final_test_data,
        'test_targets':  final_test_targets.tolist(),
        'class_names':   CLASSES_TO_KEEP,
    }, save_path)

    print(f"\n  Saved processed dataset -> {save_path}")
    print(f"  File size: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")
    print("\n  Done! [OK]\n")


if __name__ == '__main__':
    main()
