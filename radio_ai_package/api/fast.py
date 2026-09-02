import base64
import os
import io
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Import explicit model loaders from your package
from radio_ai_package.ml_logic.models.model_CNN import load_cnn_model_from_bucket
from radio_ai_package.ml_logic.models.model_vgg import load_vgg_model_from_bucket
from radio_ai_package.ml_logic.models.model_yolo import load_yolo_model_from_bucket

from radio_ai_package.ml_logic.grad_cam import (
    find_last_conv_layer,
    generate_gradcam_heatmap,
    overlay_gradcam,
)


def load_requested_model(model_choice: str):
    """Dispatcher to route loading requests to correct bucket loader function."""
    choice = model_choice.lower().strip()
    if choice == "vgg":
        return load_vgg_model_from_bucket()
    elif choice == "yolo":
        return load_yolo_model_from_bucket()
    else:
        return load_cnn_model_from_bucket()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern FastAPI startup and shutdown lifecycle management."""
    # --- STARTUP LOGIC ---
    for choice in ["cnn", "vgg", "yolo"]:
        try:
            model = load_requested_model(choice)
            setattr(app.state, f"{choice}_model", model)
            print(f"✅ {choice.upper()} Model pre-loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to pre-load {choice.upper()} Model: {e}")

    yield  # Application serves incoming requests here

    # --- SHUTDOWN LOGIC ---
    print("Shutting down Radio AI API... releasing resources.")


# Initialize app with lifespan manager (replaces deprecated @app.on_event)
app = FastAPI(title="Radio AI API", version="1.0", lifespan=lifespan)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Radio AI API is running"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    gt_file: UploadFile = File(None),
    task_type: str = Form("classification"),
    model_choice: str = Form("cnn"),
    target_mode: str = Form("fracture_only"),
):
    """Primary analysis endpoint for classification, Grad-CAM, & YOLO detection visualization."""

    # 1) Read and validate file bytes
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 2) Preprocess Image for Model Input
    try:
        image_pil = Image.open(io.BytesIO(file_bytes))

        # Prepare Grayscale Tensor for Keras (CNN / VGG)
        image_gray = image_pil.convert("L")
        img_resized = image_gray.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        input_tensor = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, 224, 224, 1)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # 3) Select Model
    model_choice_clean = model_choice.lower().strip()
    attr_name = f"{model_choice_clean}_model"
    model = getattr(app.state, attr_name, None)

    if model is None:
        try:
            model = load_requested_model(model_choice_clean)
            setattr(app.state, attr_name, model)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load requested model: {e}"
            )

    # 4) Model Prediction
    fracture_prob = 0.0
    yolo_results = None

    try:
        if model_choice_clean == "yolo":
            # Native Ultralytics YOLO inference (Requires RGB PIL image)
            image_rgb = image_pil.convert("RGB")
            yolo_results = model(image_rgb)

            # Extract classification confidence or top detection box confidence
            if hasattr(yolo_results[0], "probs") and yolo_results[0].probs is not None:
                fracture_prob = float(yolo_results[0].probs.top1conf.item())
            elif (
                hasattr(yolo_results[0], "boxes")
                and yolo_results[0].boxes is not None
                and len(yolo_results[0].boxes) > 0
            ):
                # Takes highest confidence box if detection model
                fracture_prob = float(yolo_results[0].boxes.conf.max().item())
            else:
                fracture_prob = 0.0  # No detections found

        else:
            # Keras / TensorFlow Models (CNN and VGG)
            preds = model.predict(input_tensor)

            # Unpack predictions if model returns multiple outputs as a list/tuple
            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            fracture_prob = float(preds[0][0])

        is_fracture = fracture_prob >= 0.5
        label = "Fracture" if is_fracture else "Normal"

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # 5) Heatmap / Bounding Box Overlay Generation
    gradcam_base64 = ""
    gradcam_layer = ""

    if model_choice_clean in ["cnn", "vgg"]:
        try:
            gradcam_layer = (
                "block5_conv3"
                if model_choice_clean == "vgg"
                else find_last_conv_layer(model)
            )

            heatmap = generate_gradcam_heatmap(
                model=model,
                input_tensor=input_tensor,
                last_conv_layer_name=gradcam_layer,
                target_mode=target_mode,
            )

            img_gray_2d = img_array.squeeze()
            overlay_rgb, _ = overlay_gradcam(img_gray_2d, heatmap, alpha=0.4)

            # Encode RGB image to Base64 via OpenCV BGR
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".png", overlay_bgr)
            gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

        except Exception as e:
            import traceback

            print(f"Grad-CAM Error: {e}\n{traceback.format_exc()}")
            gradcam_base64 = f"ERROR: {str(e)}"

    elif model_choice_clean == "yolo":
        try:
            gradcam_layer = "N/A (YOLO Bounding Boxes)"

            # Ultralytics results.plot() renders bounding boxes onto a BGR numpy array
            annotated_bgr = yolo_results[0].plot()

            # Resize annotated image to standard response dimensions (224, 224)
            annotated_bgr = cv2.resize(annotated_bgr, (224, 224))

            _, buffer = cv2.imencode(".png", annotated_bgr)
            gradcam_base64 = base64.b64encode(buffer).decode("utf-8")

        except Exception as e:
            print(f"YOLO Box Plotting Error: {e}")
            gradcam_base64 = f"ERROR: {str(e)}"

    # 6) Return Response JSON
    return {
        "task_type": task_type,
        "model_used": model_choice_clean,
        "filename": file.filename,
        "is_fracture": is_fracture,
        "prediction_label": label,
        "fracture_probability": round(fracture_prob, 4),
        "target_mode_used": target_mode,
        "gradcam_layer": gradcam_layer,
        "gradcam_base64": gradcam_base64,
    }
