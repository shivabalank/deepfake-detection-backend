"""
video_predict.py
------------------
Runs video inference:
    Video -> Frame Extraction (every N sec) -> Face Detection ->
    Classifier -> Majority Voting -> Final Result -> PDF Report

Colab usage:
    !python video_predict.py --video /content/drive/MyDrive/test_videos/sample.mp4
"""

import os
import time
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import cfg
from utils.logger import get_logger
from utils.face_detection import FaceDetector
from utils.preprocessing import get_eval_transforms
from utils.gradcam import GradCAM, overlay_heatmap
from utils.pdf_report import generate_report
from models.model import DeepfakeClassifier
from predict import load_model

logger = get_logger(__name__)


def extract_frames(video_path: str, interval_sec: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, int(round(fps * interval_sec)))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    logger.info(f"Extracted {len(frames)} frames (every {interval_sec}s) from {video_path}")
    return frames


def predict_video(video_path: str, interval_sec: float = None):
    t0 = time.time()
    interval_sec = interval_sec or cfg.FRAME_SAMPLE_INTERVAL_SEC

    model = load_model()
    detector = FaceDetector()
    transform = get_eval_transforms()

    frames = extract_frames(video_path, interval_sec)
    if len(frames) == 0:
        raise RuntimeError("No frames extracted from video.")

    target_layer = model.get_last_conv_layer()
    cam_engine = GradCAM(model, target_layer)

    fake_count, real_count = 0, 0
    frame_probs = []
    best_fake_frame = None  # (prob, face_rgb, cam)
    best_conf = -1.0

    for frame in frames:
        face_rgb, _found = detector.detect_and_crop(frame)
        input_tensor = transform(face_rgb).unsqueeze(0).to(cfg.DEVICE)
        cam, prob = cam_engine.generate(input_tensor)
        frame_probs.append(prob)

        if prob >= 0.5:
            fake_count += 1
        else:
            real_count += 1

        # keep the most confident "fake" frame for the report's Grad-CAM visual
        if prob > best_conf:
            best_conf = prob
            face_resized = cv2.resize(face_rgb, (cfg.IMG_SIZE, cfg.IMG_SIZE))
            best_fake_frame = (prob, face_resized, cam)

    total_frames = len(frames)
    final_prediction = "FAKE" if fake_count > real_count else "REAL"
    final_confidence = (max(fake_count, real_count) / total_frames) * 100

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(cfg.OUTPUTS_DIR, base_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save Grad-CAM visuals for the most representative frame
    _, face_resized, cam = best_fake_frame
    overlay = overlay_heatmap(face_resized, cam)
    original_path = os.path.join(out_dir, "original.png")
    heatmap_path = os.path.join(out_dir, "heatmap.png")
    overlay_path = os.path.join(out_dir, "overlay.png")
    cv2.imwrite(original_path, cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR))
    cv2.imwrite(heatmap_path, cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET))
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Bar chart: Fake vs Real frame counts
    chart_path = os.path.join(out_dir, "frame_stats_chart.png")
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["Real", "Fake"], [real_count, fake_count],
                   color=["#27ae60", "#c0392b"])
    ax.set_ylabel("Frame Count")
    ax.set_title("Fake vs Real Frame Counts")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                 str(int(b.get_height())), ha="center")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    processing_time = time.time() - t0

    frame_stats = {
        "Total Frames Analyzed": total_frames,
        "Fake Frames": fake_count,
        "Real Frames": real_count,
        "Fake Frame %": f"{(fake_count / total_frames) * 100:.1f}%",
    }

    logger.info(f"Video result: {final_prediction} ({final_confidence:.2f}%) | "
                f"{frame_stats} | time={processing_time:.2f}s")

    report_path = os.path.join(cfg.REPORTS_DIR, f"{base_name}_report.pdf")
    generate_report(
        output_path=report_path,
        media_name=os.path.basename(video_path),
        prediction=final_prediction,
        confidence=final_confidence,
        model_used=cfg.MODEL_NAME,
        processing_time_sec=processing_time,
        heatmap_paths={"original": original_path, "heatmap": heatmap_path, "overlay": overlay_path},
        frame_stats=frame_stats,
        frame_stats_chart_path=chart_path,
    )

    return {
        "prediction": final_prediction,
        "confidence": final_confidence,
        "frame_stats": frame_stats,
        "report_path": report_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--interval", type=float, default=None, help="Frame sampling interval (sec)")
    args = parser.parse_args()

    result = predict_video(args.video, args.interval)
    print(result)
