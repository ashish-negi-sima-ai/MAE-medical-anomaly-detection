"""
Convert BraTS2020 H5 files to NPY format and organize into split directories.

Each H5 file contains:
  - 'image': shape (240, 240, 4), dtype float64
  - 'mask':  shape (240, 240, 3), dtype uint8

This script:
  1. Reads each H5 file's 'image' array and saves it as a .npy file.
  2. Organizes the .npy files into val/normal, val/abnormal, test/normal,
     test/abnormal, and train/normal directories based on the split text files.

Usage:
    python convert_h5_to_npy.py
"""

from __future__ import annotations

import os
import glob
import numpy as np
import h5py
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

H5_DIR = os.path.join(BASE_DIR, "dataset", "BraTS2020_training_data", "content", "data")
SPLIT_DIR = os.path.join(BASE_DIR, "dataset", "split", "brats")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "BraTS2020_training_data", "split")

# Mapping from split text files to output subdirectories
SPLIT_MAP = {
    "normal_val_samples.txt":      os.path.join("val", "normal"),
    "abnormal_val_samples.txt":    os.path.join("val", "abnormal"),
    "normal_test_samples.txt":     os.path.join("test", "normal"),
    "abnormal_test_samples.txt":   os.path.join("test", "abnormal"),
    "normal_training_samples.txt": os.path.join("train", "normal"),
}


def load_split_filenames(split_file: str) -> list[str]:
    """Read the list of .npy filenames from a split text file."""
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def convert_single_h5(h5_path: str, npy_path: str) -> None:
    """Convert a single H5 file to NPY by extracting the 'image' key."""
    with h5py.File(h5_path, "r") as h5f:
        image = np.array(h5f["image"])  # (240, 240, 4)
    np.save(npy_path, image)


def main():
    # ── Step 1: Build a set of all filenames needed by the splits ────────────
    all_needed = {}  # npy_filename -> list of (split_name, out_dir)
    for split_file, sub_dir in SPLIT_MAP.items():
        split_path = os.path.join(SPLIT_DIR, split_file)
        if not os.path.exists(split_path):
            print(f"[WARN] Split file not found, skipping: {split_path}")
            continue
        filenames = load_split_filenames(split_path)
        for fname in filenames:
            if fname not in all_needed:
                all_needed[fname] = []
            all_needed[fname].append((split_file, sub_dir))

    print(f"Total unique .npy files needed across all splits: {len(all_needed)}")

    # ── Step 2: Create output directories ────────────────────────────────────
    for sub_dir in SPLIT_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, sub_dir), exist_ok=True)

    # ── Step 3: Convert and place files ──────────────────────────────────────
    missing = []
    converted = 0

    for npy_fname, destinations in tqdm(all_needed.items(), desc="Converting"):
        # Derive the corresponding H5 filename (replace .npy -> .h5)
        h5_fname = npy_fname.replace(".npy", ".h5")
        h5_path = os.path.join(H5_DIR, h5_fname)

        if not os.path.exists(h5_path):
            missing.append(h5_fname)
            continue

        # Convert once, then copy/save to each destination
        with h5py.File(h5_path, "r") as h5f:
            image = np.array(h5f["image"])  # (240, 240, 4)

        for _, sub_dir in destinations:
            out_path = os.path.join(OUTPUT_DIR, sub_dir, npy_fname)
            if not os.path.exists(out_path):
                np.save(out_path, image)

        converted += 1

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\nDone! Converted {converted} H5 files to NPY.")
    if missing:
        print(f"[WARN] {len(missing)} H5 files were not found:")
        for m in missing[:10]:
            print(f"  - {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more.")


if __name__ == "__main__":
    main()
