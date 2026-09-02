import base64
from contextlib import asynccontextmanager
from typing import Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf

# Import deep learning model loader functions from custom package
from radio_ai_package.ml_logic.grad_cam import (
    find_last_conv_layer,
    generate_gradcam_heatmap,
)
from radio_ai_package.ml_logic.models.model_CNN import load_model_from_bucket
from radio_ai_package.ml_logic.models.model_vgg import load_vgg_model_from_bucket
from radio_ai_package.ml_logic.models.model_yolo import load_yolo_model_from_bucket

# --- GLOBAL APP STATE & CONFIG ---
MODELS = {}
TARGET_SIZE = (224, 224)


# --- LIFESPAN MANAGER (Model Preloading & Cleanup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preloads heavy deep learning models into RAM at server startup
    and cleans up memory resources upon server shutdown.
    """
    print("Initializing server and preloading deep learning models into RAM...")
    try:
        MODELS["cnn"] = load_model_from_bucket()
        MODELS["vgg"] = load_vgg_model_from_bucket()
        MODELS["yolo"] = load_yolo_model_from_bucket()
        print("All models (CNN, VGG, YOLO) loaded successfully!")
    except Exception as e:
        print(f"Warning: Model preloading encountered an error: {e}")

    yield  # API is live and accepting incoming client requests

    print("Shutting down API server... clearing model memory.")
    MODELS.clear()


# Initialize FastAPI app with lifespan manager
app = FastAPI(
    title="GRAZPEDWRI-DX Fracture Diagnostics API",
    description="Multi-task API for X-ray Classification (CNN/VGG + Grad-CAM) and Segmentation (YOLO)",
    lifespan=lifespan,
)

# CORS Middleware to allow Streamlit frontend requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- PREPROCESSING & HELPER FUNCTIONS ---
def preprocess_image_to_tensor(
    img_gray: np.ndarray,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> tf.Tensor:
    """Converts a grayscale NumPy image array into a normalized 4D float32 input tensor.

    Returned Shape: (1, height, width, 1)
    """
    img_resized = cv2.resize(img_gray, target_size)
    img_normalized = img_resized.astype(np.float32) / 255.0

    if len(img_normalized.shape) == 2:
        img_normalized = np.expand_dims(img_normalized, axis=-1)

    input_tensor = np.expand_dims(img_normalized, axis=0)
    return tf.convert_to_tensor(input_tensor, dtype=tf.float32)


def overlay_gradcam(
    img_base_gray: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> tuple[np.ndarray, np.ndarray]:
    """Superimposes the Grad-CAM JET heatmap over the original grayscale image matrix."""
    img_resized = cv2.resize(img_base_gray, TARGET_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

    heatmap_resized = cv2.resize(heatmap, TARGET_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Convert single-channel heatmap into 3-channel JET color map
    heatmap_jet_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_jet_rgb = cv2.cvtColor(heatmap_jet_bgr, cv2.COLOR_BGR2RGB)

    # Alpha blend original image with color map
    superimposed_rgb = cv2.addWeighted(
        img_rgb, 1 - alpha, heatmap_jet_rgb, alpha, 0
    )
    return superimposed_rgb, heatmap_jet_rgb


def encode_image_to_base64(img_rgb: np.ndarray) -> str:
    """Encodes an RGB NumPy image matrix to a Base64-encoded PNG string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


def compute_iou(
    box1: list[float], box2: list[float], eps: float = 1e-6
) -> float:
    """Computes Intersection over Union (IoU) between two boxes [xmin, ymin, xmax, ymax]."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return float(intersection / (union + eps))


def parse_gt_label_file(
    content: str, img_w: int, img_h: int
) -> list[dict[str, any]]:
    """Converts normalized YOLO label text [cls, x_ctr, y_ctr, w, h] to pixel coordinates."""
    gt_boxes = []
    lines = content.strip().split("\n")
    for line in lines:
        if not line.strip():
            continue
        parts = list(map(float, line.strip().split()))
        cls_id = int(parts[0])
        x_ctr, y_ctr, bw, bh = parts[1:5]

        # Map normalized coordinates (0.0 - 1.0) to pixel bounds
        xmin = (x_ctr - bw / 2) * img_w
        ymin = (y_ctr - bh / 2) * img_h
        xmax = (x_ctr + bw / 2) * img_w
        ymax = (y_ctr + bh / 2) * img_h

        gt_boxes.append(
            {
                "class_id": cls_id,
                "box": [round(xmin, 2), round(ymin, 2), round(xmax, 2), round(ymax, 2)],
            }
        )
    return gt_boxes


# --- API ENDPOINTS ---


@app.post("/analyze")
async def analyze_xray(
    file: UploadFile = File(...),
    gt_file: Optional[UploadFile] = File(None),
    task_type: str = Form(
        ..., description="'classification' or 'segmentation'"
    ),
    model_choice: str = Form(..., description="'cnn', 'vgg', or 'yolo'"),
    target_mode: str = Form(
        "fracture_only", description="'fracture_only' or 'winning_class'"
    ),
):
    """Main multi-task routing endpoint for X-ray classification and segmentation."""
    contents = await file.read()

    # ==========================================
    # ROUTE 1: CLASSIFICATION (CNN or VGG + Grad-CAM)
    # ==========================================
    if task_type.lower() == "classification":
        if model_choice.lower() not in ["cnn", "vgg"]:
            raise HTTPException(
                status_code=400,
                detail="For classification, model_choice must be 'cnn' or 'vgg'.",
            )

        model = MODELS.get(model_choice.lower())
        if model is None:
            raise HTTPException(
                status_code=500, detail=f"Model '{model_choice}' is not loaded."
            )

        # Decode incoming byte stream into grayscale image matrix
        np_arr = np.frombuffer(contents, np.uint8)
        img_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise HTTPException(
                status_code=400, detail="Invalid image file format."
            )

        # Preprocess grayscale image array to 4D tensor (1, 224, 224, 1)
        input_tensor = preprocess_image_to_tensor(img_gray, TARGET_SIZE)

        # Run binary classification inference
        pred_prob = float(model.predict(input_tensor, verbose=0)[0][0])
        has_fracture = bool(pred_prob >= 0.5)

        # Dynamically locate target layer and generate Grad-CAM visualization
        gradcam_base64 = ""
        last_conv_layer = ""
        try:
            if model_choice.lower() == "vgg":
                last_conv_layer = "block5_conv3"
            else:
                last_conv_layer = find_last_conv_layer(model)

            heatmap = generate_gradcam_heatmap(
                model=model,
                input_tensor=input_tensor,
                last_conv_layer_name=last_conv_layer,
                target_mode=target_mode,
            )
            overlay_rgb, _ = overlay_gradcam(img_gray, heatmap, alpha=0.4)
            gradcam_base64 = encode_image_to_base64(overlay_rgb)
        except Exception as e:
            print(f"Warning: Grad-CAM generation failed: {e}")

        return {
            "task_type": "classification",
            "model_used": model_choice.lower(),
            "filename": file.filename,
            "is_fracture": has_fracture,
            "prediction_label": (
                "Fracture Detected" if has_fracture else "Normal"
            ),
            "fracture_probability": round(pred_prob, 4),
            "target_mode_used": target_mode,
            "gradcam_layer": last_conv_layer,
            "gradcam_base64": gradcam_base64,
        }

    # ==========================================
    # ROUTE 2: SEGMENTATION (YOLO Localization)
    # ==========================================
    elif task_type.lower() == "segmentation":
        if model_choice.lower() != "yolo":
            raise HTTPException(
                status_code=400,
                detail="For segmentation, model_choice must be 'yolo'.",
            )

        yolo_model = MODELS.get("yolo")
        if yolo_model is None:
            raise HTTPException(
                status_code=500, detail="YOLO model is not loaded."
            )

        # Decode byte stream into 3-channel color image matrix for YOLO
        np_arr = np.frombuffer(contents, np.uint8)
        raw_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if raw_img is None:
            raise HTTPException(
                status_code=400, detail="Invalid image file format."
            )

        img_h, img_w, _ = raw_img.shape

        # Parse Ground Truth file if sent by Streamlit
        gt_boxes = []
        if gt_file is not None:
            gt_content = (await gt_file.read()).decode("utf-8")
            gt_boxes = parse_gt_label_file(gt_content, img_w, img_h)

        # Run object detection inference
        results = yolo_model(raw_img)[0]

        # Format detection bounding boxes and calculate per-box IoU
        boxes = []
        for box in results.boxes:
            coords = [round(c, 2) for c in box.xyxy[0].tolist()]  # [xmin, ymin, xmax, ymax]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            # Calculate IoU against ground truth boxes of the same class
            best_iou = 0.0
            for gt in gt_boxes:
                if gt["class_id"] == cls_id:
                    iou = compute_iou(coords, gt["box"])
                    if iou > best_iou:
                        best_iou = iou

            boxes.append(
                {
                    "box": coords,
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "matched_iou": round(best_iou, 4),
                }
            )

        # Calculate overall mean IoU
        mean_iou = (
            round(float(np.mean([b["matched_iou"] for b in boxes])), 4)
            if (boxes and gt_boxes)
            else 0.0
        )

        # Plot predicted bounding boxes onto image and encode back to Base64
        annotated_bgr = results.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        segmented_base64 = encode_image_to_base64(annotated_rgb)

        return {
            "task_type": "segmentation",
            "model_used": "yolo",
            "filename": file.filename,
            "image_dimensions": {"width": img_w, "height": img_h},
            "ground_truth_boxes": gt_boxes,
            "detections_count": len(boxes),
            "detected_boxes": boxes,
            "mean_iou": mean_iou,
            "segmented_image_base64": segmented_base64,
        }

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid task_type. Choose either 'classification' or 'segmentation'.",
        )


@app.get("/check")
def check_status():
    """Health check endpoint to verify server status and loaded models."""
    return {
        "status": "online",
        "loaded_models": list(MODELS.keys()),
    }
