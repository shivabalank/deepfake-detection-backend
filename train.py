"""
train.py
---------
Trains the EfficientNet-B4 deepfake classifier.

Colab usage:
    !python train.py

Make sure config.py DATASET_ROOT points at your Google Drive dataset
folder (see README.md) before running this.
"""

import os
import re
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import cfg
from utils.logger import get_logger
from utils.preprocessing import get_train_transforms, get_eval_transforms
from utils.dataset import DeepfakeFaceDataset
from utils.metrics import compute_metrics, plot_confusion_matrix
from models.model import DeepfakeClassifier

logger = get_logger(__name__)

torch.manual_seed(cfg.SEED)
np.random.seed(cfg.SEED)

_CKPT_PATTERN = re.compile(r"^epoch_(\d+)_(mid|complete)\.pth$")


def find_latest_checkpoint(ckpt_dir: str):
    """
    Scans ckpt_dir for versioned checkpoints named epoch_NNN_mid.pth or
    epoch_NNN_complete.pth and returns (path, is_mid_epoch) for the most
    advanced one found, or (None, None) if none exist.

    Using a unique filename PER EPOCH (rather than one shared "mid_epoch.pth"
    / "last_epoch.pth" name) means a stray or accidental run — e.g. one that
    starts fresh at epoch 1 because it couldn't find a checkpoint — can never
    silently overwrite further-along progress from a later epoch. This is a
    direct fix for a real incident where exactly that happened.
    """
    if not os.path.isdir(ckpt_dir):
        return None, None

    candidates = []
    for fname in os.listdir(ckpt_dir):
        m = _CKPT_PATTERN.match(fname)
        if m:
            epoch_num = int(m.group(1))
            kind = m.group(2)
            # Sort key: a "complete" checkpoint for epoch N is more advanced
            # than a "mid" checkpoint for the SAME epoch N (rank 1 vs 0), but
            # a "mid" checkpoint for a LATER epoch is more advanced than a
            # "complete" checkpoint for an earlier epoch (epoch_num dominates).
            rank = (epoch_num, 0 if kind == "mid" else 1)
            candidates.append((rank, fname, kind))

    if not candidates:
        return None, None

    candidates.sort()
    _, fname, kind = candidates[-1]
    return os.path.join(ckpt_dir, fname), (kind == "mid")


def build_dataloaders():
    train_ds = DeepfakeFaceDataset(
        cfg.TRAIN_REAL_DIR, cfg.TRAIN_FAKE_DIR, transform=get_train_transforms(),
        subset_fraction=cfg.TRAIN_SUBSET_FRACTION,
    )
    val_ds = DeepfakeFaceDataset(
        cfg.VAL_REAL_DIR, cfg.VAL_FAKE_DIR, transform=get_eval_transforms(),
        subset_fraction=cfg.VAL_SUBSET_FRACTION,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    return train_ds, val_loader


def build_train_loader_for_epoch(train_ds, epoch: int, start_sample_idx: int = 0) -> DataLoader:
    """
    Creates the train DataLoader with a shuffle order that is deterministic
    given the epoch number (via a seeded generator + torch.randperm, which is
    exactly what DataLoader's internal RandomSampler uses for shuffle=True,
    so this reproduces the identical order).

    When resuming mid-epoch, `start_sample_idx` lets us build a Subset that
    starts DIRECTLY at the resume point in that same shuffle order — instead
    of iterating the full loader from the beginning and skipping already-seen
    batches. Skipping via "continue" still costs the full data-loading time
    (image read + face-detection transform) for every skipped batch, which
    for a resume near the end of a large epoch can take longer than the
    actual remaining training — this avoids that entirely.
    """
    generator = torch.Generator()
    generator.manual_seed(cfg.SEED * 1000 + epoch)
    n = len(train_ds)
    perm = torch.randperm(n, generator=generator).tolist()
    remaining_indices = perm[start_sample_idx:]
    subset = Subset(train_ds, remaining_indices)
    return DataLoader(
        subset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )


def run_epoch(model, loader, criterion, optimizer=None, desc="epoch",
              batch_offset: int = 0, checkpoint_fn=None):
    """
    Args:
        batch_offset: the loader may already start partway through an epoch
            (see build_train_loader_for_epoch's start_sample_idx). This is
            added to the loader's own local batch index so checkpoint_fn
            receives the TRUE position within the full epoch, not the
            position within just this (possibly truncated) loader.
        checkpoint_fn: optional callable(true_batch_idx) invoked periodically
            during training so the caller can save a resumable checkpoint.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    progress = tqdm(total=len(loader), desc=desc, unit="batch", leave=True)

    with torch.set_grad_enabled(is_train):
        for local_batch_idx, (images, labels) in enumerate(loader):
            images = images.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.4f}")

            true_batch_idx = batch_offset + local_batch_idx
            if is_train and checkpoint_fn is not None and cfg.CHECKPOINT_EVERY_N_BATCHES > 0:
                if (true_batch_idx + 1) % cfg.CHECKPOINT_EVERY_N_BATCHES == 0:
                    checkpoint_fn(true_batch_idx + 1)

    progress.close()

    n_seen = max(1, len(all_labels))
    avg_loss = total_loss / n_seen
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    return avg_loss, metrics, np.array(all_labels), np.array(all_preds)


def main():
    logger.info(f"Using device: {cfg.DEVICE}")

    # ------------------------------------------------------------------
    # SAFETY GUARD: refuse to train on CPU unless explicitly forced.
    # A CPU session (e.g. if the GPU runtime failed to attach) will still
    # "work" but produce a far-less-trained model very slowly — and worse,
    # it saves checkpoints to the SAME filenames as your real GPU-trained
    # progress, silently overwriting good checkpoints with much worse ones.
    # This has already happened once — this guard stops it from happening
    # again. Set DEEPFAKE_ALLOW_CPU_TRAINING=1 to override intentionally
    # (e.g. for quick pipeline testing on a tiny dataset).
    # ------------------------------------------------------------------
    if cfg.DEVICE.type == "cpu" and os.environ.get("DEEPFAKE_ALLOW_CPU_TRAINING") != "1":
        logger.error(
            "Refusing to train on CPU (no GPU detected). This is almost always "
            "because the Colab runtime didn't attach a GPU this session. "
            "Fix: Runtime -> Change runtime type -> GPU, then Runtime -> Restart "
            "session, then re-run your setup cells. "
            "If you really intend to train on CPU anyway, set the environment "
            "variable DEEPFAKE_ALLOW_CPU_TRAINING=1 before running this script."
        )
        return

    train_ds, val_loader = build_dataloaders()

    model = DeepfakeClassifier(pretrained=True).to(cfg.DEVICE)
    model.freeze_backbone()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.LR_SCHEDULER_FACTOR,
        patience=cfg.LR_SCHEDULER_PATIENCE
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_epoch = 1
    start_batch = 0
    best_ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")

    # ------------------------------------------------------------------
    # RESUME SUPPORT: find the most advanced VERSIONED checkpoint on disk
    # (epoch_NNN_mid.pth or epoch_NNN_complete.pth). Each epoch gets its own
    # filename, so this can never be silently overwritten by an older/worse
    # run the way a single shared "last_epoch.pth"/"mid_epoch.pth" name
    # could be (this happened once already — see README for the incident).
    # ------------------------------------------------------------------
    latest_ckpt_path, is_mid = find_latest_checkpoint(cfg.CHECKPOINT_DIR)
    if latest_ckpt_path:
        logger.info(f"Found checkpoint at {latest_ckpt_path} (mid_epoch={is_mid}), resuming...")
        checkpoint = torch.load(latest_ckpt_path, map_location=cfg.DEVICE, weights_only=False)

        best_val_loss = checkpoint["best_val_loss"]
        epochs_no_improve = checkpoint["epochs_no_improve"]
        if is_mid:
            start_epoch = checkpoint["epoch"]
            start_batch = checkpoint["batch_idx"]
        else:
            start_epoch = checkpoint["epoch"] + 1
            start_batch = 0

        # IMPORTANT: if the checkpoint being resumed is at/after the
        # backbone-unfreeze epoch, the optimizer that was saved covers ALL
        # model parameters (backbone + head), not just the head. We must
        # unfreeze the backbone and recreate the optimizer with that same,
        # larger parameter set BEFORE loading its saved state — otherwise
        # optimizer.load_state_dict() fails with a parameter-group-size
        # mismatch (the freshly-created optimizer at this point in the code
        # still only knows about the frozen-backbone/head-only parameters).
        if start_epoch >= cfg.FREEZE_BACKBONE_EPOCHS + 1:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.LEARNING_RATE / 2, weight_decay=cfg.WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=cfg.LR_SCHEDULER_FACTOR,
                patience=cfg.LR_SCHEDULER_PATIENCE
            )
            already_unfrozen_at_resume = True
        else:
            already_unfrozen_at_resume = False

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Resumed at epoch {start_epoch}, batch {start_batch}, "
                    f"best_val_loss={best_val_loss:.4f}")
    else:
        logger.info("No existing checkpoint found — starting fresh from epoch 1.")
        already_unfrozen_at_resume = False

    resume_start_epoch = start_epoch  # fixed reference; start_epoch's loop use doesn't change this

    for epoch in range(start_epoch, cfg.NUM_EPOCHS + 1):
        if (epoch == cfg.FREEZE_BACKBONE_EPOCHS + 1 and start_batch == 0
                and not (epoch == resume_start_epoch and already_unfrozen_at_resume)):
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.LEARNING_RATE / 2, weight_decay=cfg.WEIGHT_DECAY
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=cfg.LR_SCHEDULER_FACTOR,
                patience=cfg.LR_SCHEDULER_PATIENCE
            )

        # Only the epoch we're actually resuming into needs to start partway
        # through; every subsequent epoch starts fresh at sample 0.
        if epoch == resume_start_epoch and start_batch > 0:
            start_sample_idx = start_batch * cfg.BATCH_SIZE
            batch_offset = start_batch
        else:
            start_sample_idx = 0
            batch_offset = 0

        train_loader = build_train_loader_for_epoch(train_ds, epoch, start_sample_idx=start_sample_idx)

        midepoch_ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"epoch_{epoch:03d}_mid.pth")
        complete_ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"epoch_{epoch:03d}_complete.pth")

        def save_midepoch(batch_idx, epoch=epoch, path=midepoch_ckpt_path):
            torch.save({
                "epoch": epoch,
                "batch_idx": batch_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "epochs_no_improve": epochs_no_improve,
            }, path)
            logger.info(f"Mid-epoch checkpoint saved: {path} (batch {batch_idx})")

        t0 = time.time()
        train_loss, train_metrics, _, _ = run_epoch(
            model, train_loader, criterion, optimizer,
            desc=f"Epoch {epoch} [train]", batch_offset=batch_offset,
            checkpoint_fn=save_midepoch,
        )

        val_loss, val_metrics, val_labels, val_preds = run_epoch(
            model, val_loader, criterion, desc=f"Epoch {epoch} [val]"
        )
        scheduler.step(val_loss)
        dt = time.time() - t0

        logger.info(
            f"Epoch {epoch}/{cfg.NUM_EPOCHS} | {dt:.1f}s | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1_score']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f}"
        )

        # Epoch finished cleanly -> that epoch's mid-checkpoint is now stale.
        if os.path.exists(midepoch_ckpt_path):
            os.remove(midepoch_ckpt_path)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": min(best_val_loss, val_loss),
            "epochs_no_improve": epochs_no_improve,
        }, complete_ckpt_path)
        logger.info(f"Epoch checkpoint saved: {complete_ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, best_ckpt_path)
            plot_confusion_matrix(
                val_labels, val_preds,
                os.path.join(cfg.REPORTS_DIR, "confusion_matrix_val.png")
            )
            logger.info(f"New best model saved to {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered.")
                break

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
