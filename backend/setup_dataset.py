"""
setup_dataset.py — PRODUCTION-GRADE DATASET BUILDER

Features:
✔ Stratified train/val/test split
✔ Detection (tumor vs no_tumor)
✔ Classification (glioma, meningioma, pituitary)
✔ Reproducible (seed control)
✔ Safe file copying (no overwrite)
✔ Detailed logging

Usage:
python setup_dataset.py --src /path/to/raw --dst dataset --val 0.1 --test 0.1
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

TUMOR_CLASSES = ["glioma", "meningioma", "pituitary"]
NO_TUMOR_CLASS = "no_tumor"

RAW_CLASS_MAP = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "notumor": "no_tumor"
}

# ─────────────────────────────────────────────────────────────
# UTIL FUNCTIONS
# ─────────────────────────────────────────────────────────────

def collect_files(src_root: Path):
    """Collect all files with labels."""
    data = []

    print(f"  Scanning: {src_root}")
    for split in ["Training", "Testing"]:
        for raw_cls, mapped_cls in RAW_CLASS_MAP.items():
            folder = src_root / split / raw_cls
            if not folder.exists():
                print(f"  [SKIP] {folder}")
                continue
            files = [f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
            print(f"  [FOUND] {folder} -> {len(files)} images")
            for file in files:
                data.append((file, mapped_cls))

    return data


def stratified_split(data, val_ratio, test_ratio):
    """Perform stratified train/val/test split."""
    paths, labels = zip(*data)

    # First split train vs temp (val+test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=SEED
    )

    # Split temp into val and test
    val_size = val_ratio / (val_ratio + test_ratio)

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_size),
        stratify=temp_labels,
        random_state=SEED
    )

    return (
        list(zip(train_paths, train_labels)),
        list(zip(val_paths, val_labels)),
        list(zip(test_paths, test_labels)),
    )


def copy_files(data, dst_root, task):
    """Copy files into detection/classification folders."""
    stats = defaultdict(int)

    for idx, (src_path, label) in enumerate(data):
        if task == "detection":
            label = "tumor" if label != "no_tumor" else "no_tumor"

        dst_dir = dst_root / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Prevent overwrite
        new_name = f"{label}_{idx}_{src_path.name}"
        shutil.copy2(src_path, dst_dir / new_name)

        stats[label] += 1

    return stats


def print_stats(name, stats):
    print(f"\n📊 {name} Distribution:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Raw dataset path")
    parser.add_argument("--dst", default="dataset", help="Output path")
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)

    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    print("\n🚀 Starting dataset setup...\n")

    # Step 1: Collect data
    data = collect_files(src_root)
    print(f"Total images found: {len(data)}")

    # Step 2: Split dataset
    train_data, val_data, test_data = stratified_split(
        data, args.val, args.test
    )

    print(f"\nSplit:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val:   {len(val_data)}")
    print(f"  Test:  {len(test_data)}")

    # Step 3: Build datasets
    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        print(f"\n📁 Processing {split_name.upper()} set")

        # Detection
        det_stats = copy_files(
            split_data,
            dst_root / "detection" / split_name,
            task="detection"
        )

        # Classification (only tumor)
        cls_data = [(p, l) for p, l in split_data if l != "no_tumor"]

        cls_stats = copy_files(
            cls_data,
            dst_root / "classification" / split_name,
            task="classification"
        )

        print_stats(f"{split_name} (Detection)", det_stats)
        print_stats(f"{split_name} (Classification)", cls_stats)

    print("\n✅ Dataset setup completed successfully!")


if __name__ == "__main__":
    main()