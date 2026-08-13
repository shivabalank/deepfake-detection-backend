"""
config.py
---------
Central configuration for the Deepfake Detection project.
Edit DATASET_ROOT to point at your Google Drive dataset location when
running in Colab (see README.md for the exact folder layout expected).
"""

import os
import torch


class Config:
    # ------------------------------------------------------------------
    # PATHS  ── THIS IS WHERE YOU POINT TO YOUR GOOGLE DRIVE DATASET ────
    # ------------------------------------------------------------------
    # In Colab, after mounting Drive with:
    #   from google.colab import drive
    #   drive.mount('/content/drive')
    #
    # Put your dataset in Drive like:
    #   /content/drive/MyDrive/deepfake_dataset/train/real/*.jpg
    #   /content/drive/MyDrive/deepfake_dataset/train/fake/*.jpg
    #   /content/drive/MyDrive/deepfake_dataset/val/real/*.jpg
    #   /content/drive/MyDrive/deepfake_dataset/val/fake/*.jpg
    #   /content/drive/MyDrive/deepfake_dataset/test/real/*.jpg
    #   /content/drive/MyDrive/deepfake_dataset/test/fake/*.jpg
    #
    # Then just change DATASET_ROOT below to that path. Nothing else
    # in the codebase needs to change.
    # Confirmed from your `find` output:
    #   deepfake_dataset_extracted/Dataset/Train/Real
    #   deepfake_dataset_extracted/Dataset/Train/Fake
    #   deepfake_dataset_extracted/Dataset/Validation/Real
    #   deepfake_dataset_extracted/Dataset/Validation/Fake
    #   deepfake_dataset_extracted/Dataset/Test/Real
    #   deepfake_dataset_extracted/Dataset/Test/Fake
    # Hardcoded directly (env-var override still works if set, but no longer required).
    DATASET_ROOT = os.environ.get(
        "DEEPFAKE_DATASET_ROOT",
        "/content/drive/MyDrive/deepfake_dataset_extracted/Dataset"
    )

    TRAIN_DIRNAME = os.environ.get("DEEPFAKE_TRAIN_DIRNAME", "Train")
    VAL_DIRNAME = os.environ.get("DEEPFAKE_VAL_DIRNAME", "Validation")
    TEST_DIRNAME = os.environ.get("DEEPFAKE_TEST_DIRNAME", "Test")

    REAL_DIRNAME = os.environ.get("DEEPFAKE_REAL_DIRNAME", "Real")
    FAKE_DIRNAME = os.environ.get("DEEPFAKE_FAKE_DIRNAME", "Fake")

    TRAIN_REAL_DIR = os.path.join(DATASET_ROOT, TRAIN_DIRNAME, REAL_DIRNAME)
    TRAIN_FAKE_DIR = os.path.join(DATASET_ROOT, TRAIN_DIRNAME, FAKE_DIRNAME)
    VAL_REAL_DIR = os.path.join(DATASET_ROOT, VAL_DIRNAME, REAL_DIRNAME)
    VAL_FAKE_DIR = os.path.join(DATASET_ROOT, VAL_DIRNAME, FAKE_DIRNAME)
    TEST_REAL_DIR = os.path.join(DATASET_ROOT, TEST_DIRNAME, REAL_DIRNAME)
    TEST_FAKE_DIR = os.path.join(DATASET_ROOT, TEST_DIRNAME, FAKE_DIRNAME)

    # Where checkpoints / reports / outputs are written.
    # In Colab these default to local disk (fast), but you can redirect
    # them into Drive too (e.g. "/content/drive/MyDrive/deepfake_ckpts")
    # so they survive a runtime restart.
    CHECKPOINT_DIR = os.environ.get("DEEPFAKE_CKPT_DIR", "./checkpoints")
    REPORTS_DIR = os.environ.get("DEEPFAKE_REPORTS_DIR", "./reports")
    OUTPUTS_DIR = os.environ.get("DEEPFAKE_OUTPUTS_DIR", "./outputs")

    # ------------------------------------------------------------------
    # MODEL
    # ------------------------------------------------------------------
    MODEL_NAME = "efficientnet_b4"
    IMG_SIZE = 380          # EfficientNet-B4 native input resolution
    NUM_CLASSES = 1         # binary -> single logit + sigmoid
    FREEZE_BACKBONE_EPOCHS = 3   # freeze backbone for first N epochs

    # ------------------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------------------
    BATCH_SIZE = 16

    # ------------------------------------------------------------------
    # REALITY CHECK ON TRAINING TIME:
    # Your full dataset (~140k train images) takes ~3 hours per epoch on a
    # free-tier Colab T4. Free-tier GPU quota is limited and resets
    # unpredictably, so 30 epochs (~90 hours of GPU compute) can take WEEKS
    # of real time fighting quota limits and session drops. If you need a
    # working model soon (e.g. to ship a demo/UI), strongly consider:
    #   - Lowering NUM_EPOCHS (10-15 is often enough with early stopping)
    #   - Lowering EARLY_STOPPING_PATIENCE (stops sooner once it plateaus)
    #   - Setting TRAIN_SUBSET_FRACTION / VAL_SUBSET_FRACTION below 1.0 to
    #     train on a fraction of the data per run, cutting epoch time
    #     roughly proportionally (e.g. 0.3 -> ~1hr/epoch instead of ~3hr)
    # A model trained on a well-chosen subset for fewer epochs is very
    # often good enough for a working demo, and you can always continue
    # training longer later once you have GPU time to spare.
    # ------------------------------------------------------------------
    NUM_EPOCHS = int(os.environ.get("DEEPFAKE_NUM_EPOCHS", 12))
    EARLY_STOPPING_PATIENCE = int(os.environ.get("DEEPFAKE_EARLY_STOPPING_PATIENCE", 3))

    TRAIN_SUBSET_FRACTION = float(os.environ.get("DEEPFAKE_TRAIN_SUBSET_FRACTION", 1.0))
    VAL_SUBSET_FRACTION = float(os.environ.get("DEEPFAKE_VAL_SUBSET_FRACTION", 1.0))

    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 2          # Colab: keep this low (2-4)
    LR_SCHEDULER_PATIENCE = 2
    LR_SCHEDULER_FACTOR = 0.5

    # Save a resumable checkpoint every N training batches (not just once per
    # epoch). Large datasets can take longer to finish one epoch than a
    # Colab session survives, so this lets you resume mid-epoch instead of
    # losing all progress since the last full epoch. ~200 batches at
    # batch_size=16 is ~3200 images between saves — tune lower if your
    # sessions disconnect very frequently, higher if checkpoint I/O to
    # Drive is slowing training down noticeably.
    CHECKPOINT_EVERY_N_BATCHES = 200

    # ImageNet normalization stats (required for pretrained EfficientNet)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ------------------------------------------------------------------
    # FACE DETECTION
    # ------------------------------------------------------------------
    FACE_MARGIN = 20          # pixels of margin around detected face box
    MIN_FACE_SIZE = 40

    # ------------------------------------------------------------------
    # VIDEO INFERENCE
    # ------------------------------------------------------------------
    FRAME_SAMPLE_INTERVAL_SEC = 1.0  # extract 1 frame per second

    # ------------------------------------------------------------------
    # MISC
    # ------------------------------------------------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    LOG_LEVEL = "INFO"

    @classmethod
    def make_dirs(cls):
        for d in [cls.CHECKPOINT_DIR, cls.REPORTS_DIR, cls.OUTPUTS_DIR]:
            os.makedirs(d, exist_ok=True)


cfg = Config()
cfg.make_dirs()
