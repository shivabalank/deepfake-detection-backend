# Deepfake Detection System (EfficientNet-B4 + Grad-CAM + PDF Reports)

Production-style deepfake detector for images and videos: face detection (MTCNN) →
EfficientNet-B4 classifier → Grad-CAM explainability → auto-generated PDF report.

---

## 1. Running this in Google Colab — Step by Step

### Step 1: Open a new Colab notebook and set GPU runtime
`Runtime -> Change runtime type -> Hardware accelerator -> GPU (T4 or better)`

### Step 2: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Upload this project to Drive (or Colab disk) and cd into it
Easiest: zip this `deepfake_detection/` folder, upload it to your Drive, then:
```python
!unzip -q "/content/drive/MyDrive/deepfake_detection.zip" -d /content/
%cd /content/deepfake_detection
```
(Or just upload the folder directly into `/content/deepfake_detection` via the Colab file browser.)

### Step 4: Install dependencies
```python
!pip install -q -r requirements.txt
```

### Step 5: Put your dataset in Google Drive — **this is the important part**

Create this exact folder structure inside **your Google Drive**:

```
MyDrive/
└── deepfake_dataset/
    ├── train/
    │   ├── real/     <-- put your REAL training face images here (.jpg/.png)
    │   └── fake/     <-- put your FAKE training face images here
    ├── val/
    │   ├── real/     <-- REAL validation images
    │   └── fake/     <-- FAKE validation images
    └── test/
        ├── real/     <-- REAL test images
        └── fake/     <-- FAKE test images
```

So the full paths Colab will read from are:
```
/content/drive/MyDrive/deepfake_dataset/train/real/
/content/drive/MyDrive/deepfake_dataset/train/fake/
/content/drive/MyDrive/deepfake_dataset/val/real/
/content/drive/MyDrive/deepfake_dataset/val/fake/
/content/drive/MyDrive/deepfake_dataset/test/real/
/content/drive/MyDrive/deepfake_dataset/test/fake/
```

**You only need to edit ONE line** — `DATASET_ROOT` in `config.py`:

```python
DATASET_ROOT = "/content/drive/MyDrive/deepfake_dataset"
```

If your dataset already lives somewhere else in Drive (e.g. you have one big
`real_vs_fake` folder with a different name, or it's inside a shared/team
Drive), just change that single path — everything else (`train/real`,
`train/fake`, etc. subfolders) is derived automatically. You don't need to
touch `dataset.py`, `train.py`, or any other file.

> Tip: you can also skip editing `config.py` and instead set an environment
> variable before running any script, which overrides the default:
> ```python
> import os
> os.environ["DEEPFAKE_DATASET_ROOT"] = "/content/drive/MyDrive/deepfake_dataset"
> ```

### Step 6: Train
```python
!python train.py
```
Checkpoints are saved to `./checkpoints/best_model.pth` (local Colab disk by
default — copy it to Drive if you want it to survive a runtime reset):
```python
!cp checkpoints/best_model.pth /content/drive/MyDrive/deepfake_ckpts/
```
Or set `DEEPFAKE_CKPT_DIR` env var to a Drive path before training so it
saves there directly.

### Step 7: Evaluate on the test set
```python
!python test.py
```

### Step 8: Run inference on a single image
```python
!python predict.py --image "/content/drive/MyDrive/test_images/sample.jpg"
```
This writes:
- `outputs/sample/original.png`, `heatmap.png`, `overlay.png`
- `reports/sample_report.pdf`

### Step 9: Run inference on a video
```python
!python video_predict.py --video "/content/drive/MyDrive/test_videos/sample.mp4"
```

### Step 10 (optional): Run the Flask API in Colab
```python
!pip install -q flask-ngrok
!python app.py
```
Uncomment the `run_with_ngrok(app)` line in `app.py` first to get a public URL.

---

## 2. Folder Structure

```
deepfake_detection/
├── dataset/                 # (local mirror — not used when DATASET_ROOT points to Drive)
├── models/
│   └── model.py             # EfficientNet-B4 + custom head
├── utils/
│   ├── preprocessing.py     # transforms/augmentation
│   ├── face_detection.py    # MTCNN wrapper
│   ├── dataset.py           # PyTorch Dataset
│   ├── gradcam.py           # Grad-CAM
│   ├── pdf_report.py        # ReportLab PDF generation
│   ├── metrics.py           # accuracy/precision/recall/F1/AUC/confusion matrix
│   └── logger.py
├── checkpoints/             # saved model weights
├── reports/                 # generated PDF reports
├── outputs/                 # heatmaps, overlays, charts
├── config.py                # <-- EDIT DATASET_ROOT HERE
├── train.py
├── test.py
├── predict.py
├── video_predict.py
├── app.py
└── requirements.txt
```

---

## 3. Model & Pipeline Summary

- **Face detection**: MTCNN crops the face (with margin) before classification;
  falls back to a center crop if no face is found so the pipeline never crashes.
- **Classifier**: EfficientNet-B4 (ImageNet-pretrained), backbone frozen for the
  first `FREEZE_BACKBONE_EPOCHS` epochs, then fine-tuned end-to-end. Custom head:
  Dropout → Linear(1792→256) → ReLU → Dropout → Linear(256→1) → sigmoid at inference.
- **Loss/optimizer**: `BCEWithLogitsLoss`, `AdamW` (lr=1e-4), `ReduceLROnPlateau`,
  early stopping, best-checkpoint saving.
- **Explainability**: Grad-CAM on the last EfficientNet block, rendered as
  original / heatmap / overlay images.
- **Video**: samples 1 frame/sec (configurable), classifies each face crop,
  majority-votes across frames, plots a Fake-vs-Real bar chart.
- **Reports**: one PDF per prediction with prediction, confidence, model,
  processing time, Grad-CAM images, frame stats (video), and conclusion text.

---

## 4. Suggestions for Improving Accuracy / Reducing False Positives

1. **Balance and diversify your dataset** — use multiple deepfake generation
   methods (FaceSwap, DeepFaceLab, StyleGAN, Diffusion-based) so the model
   doesn't overfit to one generator's artifacts.
2. **Frame-level to identity-level aggregation for video** — instead of simple
   majority voting, weight votes by detector confidence, or use temporal
   models (e.g. an LSTM/Transformer over frame embeddings).
3. **Add frequency-domain features** — deepfakes often leave artifacts in the
   frequency spectrum (DCT/FFT); a secondary branch analyzing these can catch
   fakes the RGB-only CNN misses.
4. **Cross-dataset validation** — test on datasets you didn't train on
   (e.g. train on FaceForensics++, validate on Celeb-DF) to catch overfitting
   to a specific dataset's compression/artifacts.
5. **Calibrate the confidence score** — apply temperature scaling or Platt
   scaling post-training so "98% confidence" is actually well-calibrated,
   reducing overconfident false positives.
6. **Use an ensemble** — combine EfficientNet-B4 with a second architecture
   (e.g. Xception, commonly used in FaceForensics++ baselines) and average
   predictions.
7. **Hard-negative mining** — after initial training, find real images the
   model misclassifies as fake (false positives) and oversample similar
   examples in the next training round.
8. **Compression-aware augmentation** — apply JPEG/H.264-style compression
   augmentation during training, since deepfakes are often shared through
   compressed platforms (social media), which changes artifact visibility.
