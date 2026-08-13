"""
utils/face_detection.py
------------------------
Wraps MTCNN (from facenet-pytorch) to detect and crop faces before they
are passed to the classifier. Falls back gracefully to a center-crop
of the full frame if no face is detected, so the pipeline never crashes
on a difficult frame.
"""

from typing import Optional, Tuple
import numpy as np
import cv2

try:
    from facenet_pytorch import MTCNN
except ImportError as e:
    raise ImportError(
        "facenet-pytorch is required. Install with: pip install facenet-pytorch"
    ) from e

from config import cfg
from utils.logger import get_logger

logger = get_logger(__name__)


class FaceDetector:
    """Thin, reusable wrapper around MTCNN face detection + cropping."""

    def __init__(self, device: Optional[str] = None, margin: int = None):
        self.device = device or cfg.DEVICE
        self.margin = margin if margin is not None else cfg.FACE_MARGIN
        self.detector = MTCNN(
            keep_all=False,           # only the most confident face
            device=self.device,
            min_face_size=cfg.MIN_FACE_SIZE,
            post_process=False,
        )

    def detect_and_crop(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Args:
            image_bgr: image as read by cv2 (BGR, HxWx3)
        Returns:
            (cropped_face_rgb, face_found)
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            boxes, probs = self.detector.detect(rgb)
        except Exception as e:
            logger.warning(f"MTCNN detection failed: {e}")
            boxes = None

        h, w = rgb.shape[:2]

        if boxes is None or len(boxes) == 0:
            logger.debug("No face detected, falling back to center crop.")
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            crop = rgb[y0:y0 + side, x0:x0 + side]
            return crop, False

        # take the highest-confidence box
        best_idx = int(np.argmax(probs))
        x1, y1, x2, y2 = boxes[best_idx]
        x1 = max(0, int(x1) - self.margin)
        y1 = max(0, int(y1) - self.margin)
        x2 = min(w, int(x2) + self.margin)
        y2 = min(h, int(y2) + self.margin)

        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            logger.warning("Empty crop after margin adjustment, using full frame.")
            crop = rgb
            return crop, False

        return crop, True
