import base64
import os
import io
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Import logic from your package
from radio_ai_package.ml_logic.model import load_model
from radio_ai_package.ml_logic.grad_cam import (
    find_last_conv_layer,
    generate_gradcam_heatmap,
    overlay_gradcam,
)

app = FastAPI(title="Radio AI API", version="1.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Pre-warm or load default models on startup if needed."""
    try:
        app.state.cnn_model = load_model(model_choice="cnn")
        print("CNN Model loaded successfully.")
    except Exception as e:
        print(f"Failed to pre-load CNN Model: {e}")

    try:
        app.state.vgg_model = load_model(model_choice="vgg")
        print("VGG Model loaded successfully.")
    except Exception as e:
        print(f"Failed to pre-load VGG Model: {e}")


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
    """Primary analysis endpoint for classification & Grad-CAM visualization."""
    # 1) Read and validate file bytes
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 2) Preprocess Image for Model Input
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("L")  # Grayscale
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0

        # Reshape to (1, 224, 224, 1) tensor
        input_tensor = np.expand_dims(img_array, axis=(0, -1))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # 3) Select Model
    model_choice_clean = model_choice.lower().strip()
    if model_choice_clean == "vgg":
        model = getattr(app.state, "vgg_model", None)
        if model is None:
            model = load_model(model_choice="vgg")
            app.state.vgg_model = model
    else:
        model = getattr(app.state, "cnn_model", None)
        if model is None:
            model = load_model(model_choice="cnn")
            app.state.cnn_model = model

    if model is None:
        raise HTTPException(status_code=500, detail="Failed to load requested model.")

    # 4) Model Prediction
    try:
        preds = model.predict(input_tensor)
        fracture_prob = float(preds[0][0])
        is_fracture = fracture_prob >= 0.5
        label = "Fracture" if is_fracture else "Normal"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # 5) Grad-CAM Heatmap & Overlay Generation
    gradcam_base64 = ""
    gradcam_layer = ""

    try:
        # Identify layer
        if model_choice_clean == "vgg":
            gradcam_layer = "block5_conv3"
        else:
            gradcam_layer = find_last_conv_layer(model)

        # Generate Heatmap
        heatmap = generate_gradcam_heatmap(
            model=model,
            input_tensor=input_tensor,
            last_conv_layer_name=gradcam_layer,
            target_mode=target_mode,
        )

        # Generate Overlay (224, 224)
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
