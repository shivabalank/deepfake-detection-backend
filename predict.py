"""
predict.py
-----------
Runs single-image inference:
    Image -> Face Detection -> Classifier -> Grad-CAM -> PDF Report

Colab usage:
    !python predict.py --image /content/drive/MyDrive/test_images/sample.jpg
"""

import os
import time
import argparse
import cv2
import numpy as np
import torch
import requests

from config import cfg
from utils.logger import get_logger
from utils.face_detection import FaceDetector
from utils.preprocessing import get_eval_transforms
from utils.gradcam import GradCAM, overlay_heatmap
from utils.pdf_report import generate_report
from models.model import DeepfakeClassifier

logger = get_logger(__name__)


def _download_checkpoint_if_needed(ckpt_path: str) -> None:
    """
    If the checkpoint doesn't exist locally, download it from MODEL_URL
    (an environment variable, e.g. set on Render to a Hugging Face Hub
    direct-download link). This lets deployment platforms fetch the large
    model file at startup instead of requiring it to be committed to git.
    """
    if os.path.exists(ckpt_path):
        return

    model_url = os.environ.get("MODEL_URL")
    if not model_url:
        logger.warning(
            f"No checkpoint found at {ckpt_path} and no MODEL_URL environment "
            f"variable set — cannot auto-download. Set MODEL_URL to a direct "
            f"download link (e.g. a Hugging Face Hub 'resolve/main/...' URL)."
        )
        return

    logger.info(f"Checkpoint not found locally — downloading from {model_url} ...")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    response = requests.get(model_url, stream=True, timeout=300)
    response.raise_for_status()

    tmp_path = ckpt_path + ".part"
    total_bytes = 0
    with open(tmp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)

    os.rename(tmp_path, ckpt_path)
    logger.info(f"Downloaded checkpoint to {ckpt_path} ({total_bytes / (1024*1024):.1f} MB)")


def load_model() -> DeepfakeClassifier:
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pth")
    _download_checkpoint_if_needed(ckpt_path)
    model = DeepfakeClassifier(pretrained=(not os.path.exists(ckpt_path))).to(cfg.DEVICE)
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded trained weights from {ckpt_path}")
    else:
        logger.warning("No trained checkpoint found — using ImageNet-only weights "
                        "(predictions will not be meaningful until you train the model).")
    model.eval()
    return model


def predict_image(image_path: str, model: DeepfakeClassifier = None,
                   detector: FaceDetector = None, save_outputs: bool = True):
    t0 = time.time()
    model = model or load_model()
    detector = detector or FaceDetector()
    transform = get_eval_transforms()

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    face_rgb, face_found = detector.detect_and_crop(image_bgr)
    input_tensor = transform(face_rgb).unsqueeze(0).to(cfg.DEVICE)

    target_layer = model.get_last_conv_layer()
    cam_engine = GradCAM(model, target_layer)
    cam, prob = cam_engine.generate(input_tensor)

    prediction = "FAKE" if prob >= 0.5 else "REAL"
    confidence = (prob if prediction == "FAKE" else 1 - prob) * 100

    # Resize face_rgb to match model input for overlay consistency
    face_resized = cv2.resize(face_rgb, (cfg.IMG_SIZE, cfg.IMG_SIZE))
    overlay = overlay_heatmap(face_resized, cam)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(cfg.OUTPUTS_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    original_path = os.path.join(out_dir, "original.png")
    heatmap_path = os.path.join(out_dir, "heatmap.png")
    overlay_path = os.path.join(out_dir, "overlay.png")

    cv2.imwrite(original_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
    heatmap_color = (cam * 255).astype(np.uint8)
    cv2.imwrite(heatmap_path, cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET))
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    processing_time = time.time() - t0

    logger.info(f"Prediction: {prediction} | Confidence: {confidence:.2f}% "
                f"| face_found={face_found} | time={processing_time:.2f}s")

    report_path = None
    if save_outputs:
        report_path = os.path.join(cfg.REPORTS_DIR, f"{base_name}_report.pdf")
        generate_report(
            output_path=report_path,
            media_name=os.path.basename(image_path),
            prediction=prediction,
            confidence=confidence,
            model_used=cfg.MODEL_NAME,
            processing_time_sec=processing_time,
            heatmap_paths={
                "original": original_path,
                "heatmap": heatmap_path,
                "overlay": overlay_path,
            },
        )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "report_path": report_path,
        "original_path": original_path,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    result = predict_image(args.image)
    print(result)
