"""
prepare_dataset.py
--------------------
Handles the common case: you have ONE zip file on Google Drive
(e.g. "archive.zip") containing real/fake face images, not already
split into train/val/test.

What this script does:
  1. Unzips the archive from Drive into local Colab disk (fast I/O).
  2. Walks the extracted folder tree and auto-detects which
     subfolders are "real" and which are "fake" (handles common
     naming variants: real/fake, Real/Fake, 0_real/1_fake,
     original/manipulated, etc.)
  3. Splits each class 80/10/10 into train/val/test.
  4. Copies files into the exact structure config.py expects:
        ./dataset/train/real, ./dataset/train/fake
        ./dataset/val/real,   ./dataset/val/fake
        ./dataset/test/real,  ./dataset/test/fake
  5. Prints a summary so you can sanity-check counts before training.

Colab usage:
    from google.colab import drive
    drive.mount('/content/drive')

    !python prepare_dataset.py \
        --zip_path "/content/drive/MyDrive/archive.zip" \
        --output_dir "./dataset"

Then in config.py set:
    DATASET_ROOT = "/content/deepfake_detection/dataset"
(or wherever --output_dir resolves to)
"""

import os
import shutil
import random
import zipfile
import argparse
from typing import List, Tuple, Optional

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Folder-name keywords used to guess which extracted subfolder is which class.
REAL_KEYWORDS = ["real", "original", "pristine", "genuine", "0_real"]
FAKE_KEYWORDS = ["fake", "manipulated", "deepfake", "synthetic", "1_fake", "gan"]


def unzip_archive(zip_path: str, extract_to: str) -> str:
    os.makedirs(extract_to, exist_ok=True)
    print(f"Unzipping {zip_path} -> {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("Unzip complete.")
    return extract_to


def _matches_any(name: str, keywords: List[str]) -> bool:
    name_lower = name.lower()
    return any(k in name_lower for k in keywords)


def find_class_folders(root: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Walks `root` and returns (real_folder_path, fake_folder_path) by matching
    folder names against REAL_KEYWORDS / FAKE_KEYWORDS. Picks the folder with
    the most matching image files if there are multiple candidates.
    """
    candidates_real, candidates_fake = [], []

    for dirpath, dirnames, filenames in os.walk(root):
        folder_name = os.path.basename(dirpath)
        n_images = sum(1 for f in filenames if f.lower().endswith(IMG_EXTENSIONS))
        if n_images == 0:
            continue
        if _matches_any(folder_name, REAL_KEYWORDS):
            candidates_real.append((dirpath, n_images))
        elif _matches_any(folder_name, FAKE_KEYWORDS):
            candidates_fake.append((dirpath, n_images))

    real_folder = max(candidates_real, key=lambda x: x[1])[0] if candidates_real else None
    fake_folder = max(candidates_fake, key=lambda x: x[1])[0] if candidates_fake else None
    return real_folder, fake_folder


def list_images(folder: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(folder):
        for f in filenames:
            if f.lower().endswith(IMG_EXTENSIONS):
                files.append(os.path.join(dirpath, f))
    return files


def split_and_copy(files: List[str], label: str, output_dir: str,
                    train_ratio: float, val_ratio: float, seed: int = 42) -> None:
    random.seed(seed)
    files = files.copy()
    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }

    for split_name, split_files in splits.items():
        dest_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(dest_dir, exist_ok=True)
        for src in split_files:
            dest = os.path.join(dest_dir, os.path.basename(src))
            # avoid filename collisions across nested subfolders
            if os.path.exists(dest):
                base, ext = os.path.splitext(os.path.basename(src))
                dest = os.path.join(dest_dir, f"{base}_{random.randint(0, 999999)}{ext}")
            shutil.copy2(src, dest)
        print(f"  {label:5s} {split_name:5s}: {len(split_files)} images -> {dest_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, default=None,
                         help="Path to archive.zip (e.g. on Drive). Omit if already extracted.")
    parser.add_argument("--extracted_dir", type=str, default="./_extracted_raw",
                         help="Where to unzip to / where the already-extracted data lives.")
    parser.add_argument("--output_dir", type=str, default="./dataset",
                         help="Where to write the train/val/test structure.")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    if args.zip_path:
        unzip_archive(args.zip_path, args.extracted_dir)
        search_root = args.extracted_dir
    else:
        search_root = args.extracted_dir

    print(f"Scanning {search_root} for real/fake folders ...")
    real_folder, fake_folder = find_class_folders(search_root)

    if real_folder is None or fake_folder is None:
        print("\n!!! Could not auto-detect real/fake folders. !!!")
        print("Folders found under the archive:")
        for dirpath, dirnames, filenames in os.walk(search_root):
            n_images = sum(1 for f in filenames if f.lower().endswith(IMG_EXTENSIONS))
            if n_images > 0:
                print(f"  {dirpath}  ({n_images} images)")
        print(
            "\nNone of these matched expected keywords "
            f"(real: {REAL_KEYWORDS}, fake: {FAKE_KEYWORDS}).\n"
            "Re-run this script with the folders renamed, OR edit "
            "REAL_KEYWORDS/FAKE_KEYWORDS at the top of prepare_dataset.py "
            "to match your dataset's actual naming."
        )
        return

    print(f"Detected REAL folder: {real_folder}")
    print(f"Detected FAKE folder: {fake_folder}")

    real_files = list_images(real_folder)
    fake_files = list_images(fake_folder)
    print(f"Found {len(real_files)} real images, {len(fake_files)} fake images.")

    if len(real_files) == 0 or len(fake_files) == 0:
        print("One of the classes has zero images — aborting. Check the detected folders above.")
        return

    print(f"\nSplitting {args.train_ratio:.0%}/{args.val_ratio:.0%}/"
          f"{1 - args.train_ratio - args.val_ratio:.0%} (train/val/test) ...")
    split_and_copy(real_files, "real", args.output_dir, args.train_ratio, args.val_ratio)
    split_and_copy(fake_files, "fake", args.output_dir, args.train_ratio, args.val_ratio)

    print(f"\nDone. Dataset ready at: {os.path.abspath(args.output_dir)}")
    print(f"Set this in config.py:\n    DATASET_ROOT = \"{os.path.abspath(args.output_dir)}\"")


if __name__ == "__main__":
    main()
