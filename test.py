"""
test.py
--------
Evaluates the best saved checkpoint on the test set and reports
accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.

Colab usage:
    !python test.py
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import cfg
from utils.logger import get_logger
from utils.preprocessing import get_eval_transforms
from utils.dataset import DeepfakeFaceDataset
from utils.metrics import compute_metrics, plot_confusion_matrix
from models.model import DeepfakeClassifier

logger = get_logger(__name__)


def main():
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Run train.py first.")

    test_ds = DeepfakeFaceDataset(cfg.TEST_REAL_DIR, cfg.TEST_FAKE_DIR, transform=get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    model = DeepfakeClassifier(pretrained=False).to(cfg.DEVICE)
    checkpoint = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(cfg.DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    logger.info(f"Test metrics: {metrics}")

    plot_confusion_matrix(
        np.array(all_labels), np.array(all_preds),
        os.path.join(cfg.REPORTS_DIR, "confusion_matrix_test.png")
    )
    logger.info(f"Confusion matrix saved to {cfg.REPORTS_DIR}/confusion_matrix_test.png")


if __name__ == "__main__":
    main()
