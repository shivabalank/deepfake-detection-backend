"""
app.py
-------
Flask backend exposing:
    POST /predict/image  -> upload an image, get prediction + report
    POST /predict/video  -> upload a video, get prediction + report
    GET  /report/<name>  -> download a generated PDF report
    GET  /output/<path>  -> fetch a generated image (heatmap/overlay/chart)

Local usage (run alongside the React frontend):
    pip install -r requirements.txt
    python app.py
    -> serves on http://localhost:5000

Colab usage (uses flask-ngrok or similar to expose the port publicly):
    !pip install flask-ngrok
    !python app.py
"""

import os
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import cfg
from utils.logger import get_logger
from utils.pdf_report import build_conclusion
from predict import predict_image, load_model
from video_predict import predict_video
from utils.face_detection import FaceDetector

logger = get_logger(__name__)

app = Flask(__name__)
# Allow the React dev server (a different origin/port) to call this API.
# In production, replace "*" with your actual frontend's URL.
CORS(app)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load model & detector once at startup (avoids reloading per-request)
_model = load_model()
_detector = FaceDetector()

ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv"}


def _allowed(filename: str, allowed_set: set) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set


def _image_to_data_uri(path: str) -> str:
    """Reads a PNG file from disk and returns it as a base64 data: URI,
    so the frontend can display it directly in an <img> tag without a
    second network request."""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.route("/predict/image", methods=["POST"])
def predict_image_route():
    if "file" not in request.files:
        return jsonify({"error": "No file part 'file' in request"}), 400
    file = request.files["file"]
    if file.filename == "" or not _allowed(file.filename, ALLOWED_IMAGE_EXT):
        return jsonify({"error": "Invalid or missing image file"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        result = predict_image(save_path, model=_model, detector=_detector)
    except Exception as e:
        logger.exception("Image prediction failed")
        return jsonify({"error": str(e)}), 500

    confidence = round(result["confidence"], 2)
    conclusion = build_conclusion(result["prediction"], confidence)

    try:
        heatmap_image = _image_to_data_uri(result["overlay_path"])
    except Exception as e:
        logger.warning(f"Could not encode overlay image: {e}")
        heatmap_image = None

    report_filename = os.path.basename(result["report_path"])

    return jsonify({
        "prediction": result["prediction"],       # "REAL" or "FAKE"
        "confidence": confidence,                  # e.g. 99.98
        "conclusion": conclusion,                   # human-readable summary sentence
        "heatmap_image": heatmap_image,             # data:image/png;base64,... (Grad-CAM overlay)
        "report_url": f"{request.host_url.rstrip('/')}/report/{report_filename}",
    })


@app.route("/predict/video", methods=["POST"])
def predict_video_route():
    if "file" not in request.files:
        return jsonify({"error": "No file part 'file' in request"}), 400
    file = request.files["file"]
    if file.filename == "" or not _allowed(file.filename, ALLOWED_VIDEO_EXT):
        return jsonify({"error": "Invalid or missing video file"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        result = predict_video(save_path)
    except Exception as e:
        logger.exception("Video prediction failed")
        return jsonify({"error": str(e)}), 500

    confidence = round(result["confidence"], 2)
    conclusion = build_conclusion(result["prediction"], confidence)
    report_filename = os.path.basename(result["report_path"])

    return jsonify({
        "prediction": result["prediction"],
        "confidence": confidence,
        "conclusion": conclusion,
        "frame_stats": result["frame_stats"],
        "report_url": f"{request.host_url.rstrip('/')}/report/{report_filename}",
    })


@app.route("/report/<path:filename>", methods=["GET"])
def get_report(filename):
    return send_from_directory(cfg.REPORTS_DIR, filename, as_attachment=True)


@app.route("/output/<path:filepath>", methods=["GET"])
def get_output(filepath):
    return send_from_directory(cfg.OUTPUTS_DIR, filepath)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(cfg.DEVICE)})


if __name__ == "__main__":
    # In Colab, use flask-ngrok or pyngrok to expose this publicly, e.g.:
    #   from flask_ngrok import run_with_ngrok
    #   run_with_ngrok(app)
    app.run(host="0.0.0.0", port=5000, debug=False)
