"""
utils/preprocessing.py
------------------------
Defines torchvision transform pipelines for train/val/test.
Train pipeline includes the augmentations requested (flip, rotation,
brightness/contrast, blur, noise, crop, color jitter). Val/test use
only deterministic resize + normalize.
"""

import numpy as np
import torch
from torchvision import transforms
from config import cfg


class GaussianNoise:
    """Adds Gaussian noise to a tensor image (after ToTensor, before Normalize)."""

    def __init__(self, mean: float = 0.0, std: float = 0.03):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


def get_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg.IMG_SIZE + 32, cfg.IMG_SIZE + 32)),
        transforms.RandomCrop((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02
        ),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.3
        ),
        transforms.ToTensor(),
        GaussianNoise(std=0.02),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Used for validation, test, and single-image/video inference."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.MEAN, std=cfg.STD),
    ])


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverses ImageNet normalization for visualization (returns HxWx3 uint8)."""
    mean = torch.tensor(cfg.MEAN).view(3, 1, 1)
    std = torch.tensor(cfg.STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)
