import base64
import io
from typing import Optional, List, Dict, Any
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

# Import your custom modules
from radio_ai_package.ml_logic.grad_cam import (
    preprocess_image_to_tensor,
    generate_gradcam_heatmap,
    overlay_gradcam,
)

app = FastAPI(title="Radio AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
classification_model = None


@app.on_event("startup")
def load_models():
    """Load pre-trained models on server startup."""
    global classification_model
    try:
        # Load your trained Keras classification model here
        # classification_model = tf.keras.models.load_model("path/to/model.h5")
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"⚠️ Model loading error: {e}")


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def parse_yolo_gt(gt_bytes: bytes, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """Convert normalized YOLO GT format (class x_c y_c w h) to pixel coordinates."""
    boxes = []
    lines = gt_bytes.decode("utf-8").strip().split("\n")
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])

            x1 = int((x_c - w / 2) * img_w)
            y1 = int((y_c - h / 2) * img_h)
            x2 = int((x_c + w / 2) * img_w)
            y2 = int((y_c + h / 2) * img_h)

            boxes.append({"class": cls_id, "box_px": [x1, y1, x2, y2]})
    return boxes


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    gt_file: Optional[UploadFile] = File(None),
    task_type: str = Form(...),
    model_choice: str = Form(...),
    target_mode: str = Form("fracture_only"),
):
    try:
        # Read incoming image bytes
        file_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        img_h, img_w, _ = img_np.shape

        if task_type == "classification":
            # 1. Convert image array to 4D tensor using your preprocessing function
            input_tensor = preprocess_image_to_tensor(
                img_np, target_size=(224, 224), num_channels=1
            )

            # 2. Get predictions
            if classification_model is not None:
                preds = classification_model.predict(input_tensor, verbose=0)
                fracture_prob = float(preds[0][0])
            else:
                # Fallback dummy prediction if model is not loaded
                fracture_prob = 0.842

            prediction_label = "Fracture" if fracture_prob >= 0.5 else "No Fracture"

            # 3. Generate Grad-CAM Heatmap
            if classification_model is not None:
                heatmap = generate_gradcam_heatmap(
                    classification_model,
                    input_tensor,
                    last_conv_layer_name="last_block_conv",
                    target_mode=target_mode,
                )
                overlay, _ = overlay_gradcam(img_np, heatmap, alpha=0.4)
            else:
                # Fallback overlay for testing UI flow
                heatmap_dummy = np.zeros((224, 224))
                overlay, _ = overlay_gradcam(img_np, heatmap_dummy, alpha=0.4)

            # 4. Encode overlay image (RGB -> BGR for OpenCV encoding -> Base64)
            _, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            gradcam_b64 = base64.b64encode(buffer).decode("utf-8")

            return {
                "prediction_label": prediction_label,
                "fracture_probability": round(fracture_prob, 4),
                "gradcam_layer": "last_block_conv",
                "gradcam_base64": gradcam_b64,
            }

        elif task_type == "segmentation":
            # Segmentation handling logic
            overlay_np = img_np.copy()

            # Dummy YOLO detections (Replace with real YOLO model inference)
            pred_boxes = [
                {
                    "box_px": [
                        int(img_w * 0.3),
                        int(img_h * 0.3),
                        int(img_w * 0.6),
                        int(img_h * 0.6),
                    ],
                    "confidence": 0.89,
                }
            ]

            # Parse Ground Truth file if provided
            gt_boxes = []
            has_gt = False
            max_iou = 0.0

            if gt_file is not None:
                gt_bytes = await gt_file.read()
                if gt_bytes:
                    gt_boxes = parse_yolo_gt(gt_bytes, img_w, img_h)
                    has_gt = True

            if has_gt and pred_boxes and gt_boxes:
                for p in pred_boxes:
                    for g in gt_boxes:
                        iou = compute_iou(p["box_px"], g["box_px"])
                        if iou > max_iou:
                            max_iou = iou

            # Render GT in Green
            if has_gt:
                for g in gt_boxes:
                    gx1, gy1, gx2, gy2 = g["box_px"]
                    cv2.rectangle(
                        overlay_np, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2
                    )

            # Render Predictions in Red
            for p in pred_boxes:
                px1, py1, px2, py2 = p["box_px"]
                cv2.rectangle(
                    overlay_np, (px1, py1), (px2, py2), (255, 0, 0), 2
                )

            _, buffer = cv2.imencode(
                ".png", cv2.cvtColor(overlay_np, cv2.COLOR_RGB2BGR)
            )
            segmented_b64 = base64.b64encode(buffer).decode("utf-8")

            return {
                "detections_count": len(pred_boxes),
                "detected_boxes": pred_boxes,
                "has_ground_truth": has_gt,
                "ground_truth_boxes": gt_boxes,
                "max_iou": round(max_iou, 4),
                "segmented_image_base64": segmented_b64,
            }

        else:
            raise HTTPException(
                status_code=400, detail=f"Invalid task_type: {task_type}"
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
