"""
utils/dataset.py
------------------
PyTorch Dataset for loading Real/Fake face images from disk (local path
or a mounted Google Drive path — same code works for both).

Expected folder layout (see config.py DATASET_ROOT):
    <root>/train/real/*.jpg
    <root>/train/fake/*.jpg
    <root>/val/real/*.jpg
    <root>/val/fake/*.jpg
    <root>/test/real/*.jpg
    <root>/test/fake/*.jpg

Label convention: real = 0, fake = 1
"""

import os
import glob
import random
from typing import Callable, List, Tuple, Optional

import cv2
import torch
from torch.utils.data import Dataset

from utils.face_detection import FaceDetector
from utils.logger import get_logger

logger = get_logger(__name__)

IMG_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def _list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        logger.warning(f"Directory not found: {folder}")
        return []
    files = []
    for ext in IMG_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, ext.upper())))
    return sorted(files)


class DeepfakeFaceDataset(Dataset):
    """
    Loads images from a `real_dir` (label 0) and `fake_dir` (label 1),
    runs MTCNN face detection + crop, then applies the given transform.
    """

    def __init__(
        self,
        real_dir: str,
        fake_dir: str,
        transform: Optional[Callable] = None,
        use_face_detection: bool = True,
        subset_fraction: float = 1.0,
        seed: int = 42,
    ):
        """
        Args:
            subset_fraction: use only this fraction (0 < f <= 1.0) of the
                available images per class. Useful to shrink epoch time to
                fit within a realistic Colab free-tier GPU session/quota
                instead of requiring ~90+ hours of total GPU compute across
                many days for the full dataset. Sampling is done AFTER
                listing all files, with a fixed seed, so it's reproducible
                across sessions (important for the deterministic per-epoch
                shuffle used for mid-epoch resume).
        """
        self.transform = transform
        self.use_face_detection = use_face_detection
        # IMPORTANT: always run face detection on CPU here. This Dataset is
        # used inside a DataLoader with num_workers > 0, which spawns worker
        # subprocesses via fork on Linux/Colab. CUDA cannot be re-initialized
        # inside a forked subprocess, so running MTCNN on cfg.DEVICE (cuda)
        # here would fail in every worker with "Cannot re-init CUDA in forked
        # subprocess" and silently fall back to a center-crop for every image.
        # The GPU is reserved for the classifier model itself in the main
        # process; face cropping is cheap enough on CPU.
        self.detector = FaceDetector(device="cpu") if use_face_detection else None

        real_files = _list_images(real_dir)
        fake_files = _list_images(fake_dir)

        if len(real_files) == 0 and len(fake_files) == 0:
            raise RuntimeError(
                f"No images found in either '{real_dir}' or '{fake_dir}'. "
                f"Check that your Google Drive dataset path in config.py is correct."
            )

        logger.info(f"Loaded {len(real_files)} real and {len(fake_files)} fake images "
                    f"from {real_dir} / {fake_dir}")

        if not (0 < subset_fraction <= 1.0):
            raise ValueError("subset_fraction must be in (0, 1.0]")

        if subset_fraction < 1.0:
            rng = random.Random(seed)
            real_files = sorted(real_files)
            fake_files = sorted(fake_files)
            rng.shuffle(real_files)
            rng.shuffle(fake_files)
            n_real = max(1, int(len(real_files) * subset_fraction))
            n_fake = max(1, int(len(fake_files) * subset_fraction))
            real_files = real_files[:n_real]
            fake_files = fake_files[:n_fake]
            logger.info(f"Using subset_fraction={subset_fraction} -> "
                        f"{len(real_files)} real, {len(fake_files)} fake images this run")

        self.samples: List[Tuple[str, int]] = (
            [(f, 0) for f in real_files] + [(f, 1) for f in fake_files]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = cv2.imread(path)
        if image is None:
            logger.warning(f"Failed to read image {path}, skipping (returning zeros).")
            image = 0  # placeholder handled below
            face = torch.zeros(3, 380, 380)
            return face, torch.tensor(label, dtype=torch.float32)

        if self.use_face_detection:
            face_rgb, _found = self.detector.detect_and_crop(image)
        else:
            face_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            face_tensor = self.transform(face_rgb)
        else:
            face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0

        return face_tensor, torch.tensor(label, dtype=torch.float32)
