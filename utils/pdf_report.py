"""
utils/pdf_report.py
---------------------
Generates a clean, professional PDF report for a single image or video
prediction, using ReportLab.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from utils.logger import get_logger

logger = get_logger(__name__)


def build_conclusion(prediction: str, confidence: float) -> str:
    """Public: builds the same conclusion sentence used in the PDF report,
    so other callers (e.g. the API) can reuse identical wording."""
    if prediction.upper() == "FAKE":
        return (
            f"The uploaded media has a high probability of being AI-generated or "
            f"manipulated. The model detected suspicious facial artifacts consistent "
            f"with deepfake generation, with a confidence of {confidence:.2f}%."
        )
    return (
        f"The uploaded media shows no significant signs of facial manipulation. "
        f"The model classified it as authentic with a confidence of {confidence:.2f}%."
    )


# Kept for internal backward-compatibility with any code still importing the
# old private name.
_build_conclusion = build_conclusion


def generate_report(
    output_path: str,
    media_name: str,
    prediction: str,
    confidence: float,
    model_used: str,
    processing_time_sec: float,
    heatmap_paths: Optional[Dict[str, str]] = None,   # {"original":..,"heatmap":..,"overlay":..}
    frame_stats: Optional[Dict[str, Any]] = None,       # video-only
    frame_stats_chart_path: Optional[str] = None,       # video-only bar chart
) -> str:
    """Builds the PDF and returns the output path."""

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontSize=22,
        textColor=colors.HexColor("#1a1a2e"), spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontSize=11,
        textColor=colors.HexColor("#555555"), spaceAfter=20,
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"], fontSize=14,
        textColor=colors.HexColor("#0f3460"), spaceBefore=14, spaceAfter=8,
    )
    body_style = styles["Normal"]

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
    )
    story = []

    # --- Header ---
    story.append(Paragraph("Deepfake Detection Report", title_style))
    story.append(Paragraph(
        "AI-Powered Facial Manipulation Analysis", subtitle_style
    ))

    # --- Prediction banner ---
    pred_color = colors.HexColor("#c0392b") if prediction.upper() == "FAKE" else colors.HexColor("#27ae60")
    banner_data = [[f"PREDICTION: {prediction.upper()}", f"CONFIDENCE: {confidence:.2f}%"]]
    banner = Table(banner_data, colWidths=[8 * cm, 8 * cm])
    banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), pred_color),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 14),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(banner)
    story.append(Spacer(1, 16))

    # --- Metadata table ---
    meta_rows = [
        ["Media Name", media_name],
        ["Date & Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Model Used", model_used],
        ["Processing Time", f"{processing_time_sec:.2f} sec"],
    ]
    meta_table = Table(meta_rows, colWidths=[5 * cm, 11 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f2f2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 16))

    # --- Grad-CAM visuals ---
    if heatmap_paths:
        story.append(Paragraph("Explainability: Grad-CAM Analysis", heading_style))
        images_row = []
        labels_row = []
        for label, path in [
            ("Original", heatmap_paths.get("original")),
            ("Grad-CAM Heatmap", heatmap_paths.get("heatmap")),
            ("Overlay", heatmap_paths.get("overlay")),
        ]:
            if path and os.path.exists(path):
                images_row.append(RLImage(path, width=4.8 * cm, height=4.8 * cm))
                labels_row.append(Paragraph(label, body_style))
        if images_row:
            vis_table = Table([images_row, labels_row], colWidths=[5 * cm] * len(images_row))
            vis_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            story.append(vis_table)
        story.append(Spacer(1, 12))

    # --- Video frame statistics ---
    if frame_stats:
        story.append(Paragraph("Frame-Level Statistics (Video)", heading_style))
        stats_rows = [["Metric", "Value"]] + [[k, str(v)] for k, v in frame_stats.items()]
        stats_table = Table(stats_rows, colWidths=[8 * cm, 8 * cm])
        stats_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f3460")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ]))
        story.append(stats_table)
        story.append(Spacer(1, 12))

        if frame_stats_chart_path and os.path.exists(frame_stats_chart_path):
            story.append(RLImage(frame_stats_chart_path, width=12 * cm, height=7 * cm))
            story.append(Spacer(1, 12))

    # --- Conclusion ---
    story.append(Paragraph("Final Conclusion", heading_style))
    story.append(Paragraph(_build_conclusion(prediction, confidence), body_style))

    doc.build(story)
    logger.info(f"PDF report generated at {output_path}")
    return output_path
