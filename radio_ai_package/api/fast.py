import base64
import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# Import OUR visualization modules
# from radio_ai_package.ml_logic.models.model_CNN import load_model_from_bucket

app = FastAPI(title="X-Ray Fracture Detection API with Grad-CAM")

# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
# app.state.model = load_model_from_bucket()
# $WIPE_END

# Enable CORS for frontend interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
#    MODEL_PATH = "models/fracture_detection_model.keras"
#    TARGET_SIZE = (224, 224)

# Define the target layer for Grad-CAM.
# We hardcode it here instead of auto-detecting for stability in API.
LAST_CONV_LAYER_NAME = "conv2d_2" # Example: Update to match your model summary

# --- STATE MANAGEMENT: Lifespan ---
# @app.on_event("startup")
# def startup_load_model():
#     """Load model once on startup."""
#     try:
#         app.state.model = tf.keras.models.load_model(MODEL_PATH)
#         print("Model loaded successfully!")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         app.state.model = None

# # --- Helper: image processing ---
# def preprocess_for_inference(contents: bytes):
#     """Converts uploaded bytes to 4D model input tensor."""
#     # ... previous implementation ...
#     return input_tensor

# def preprocess_for_gradcam(contents: bytes):
#     """Converts uploaded bytes to raw image (H, W, 1) and raw overlay (H, W, 3)."""
#     # 1. Standard Gray Input (like before)
#     np_arr = np.frombuffer(contents, np.uint8)
#     img_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
#     if img_gray is None: raise ValueError("Invalid image file format.")
#     img_resized = cv2.resize(img_gray, TARGET_SIZE)
#     img_norm = img_resized.astype(np.float32) / 255.0
#     input_tensor = np.expand_dims(np.expand_dims(img_norm, axis=-1), axis=0) # (1, 224, 224, 1)

#     # 2. For visualization overlay: Need a color base (H, W, 3)
#     img_rgb_base = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
#     img_rgb_base_resized = cv2.resize(img_rgb_base, TARGET_SIZE)

#     return input_tensor, img_rgb_base_resized

# # --- ENDPOINT: Predict & Grad-CAM ---
# @app.post("/predict_with_gradcam")
# async def predict_with_gradcam(file: UploadFile = File(...)):
#     """Accepts an uploaded X-ray image (PNG/JPG/JPEG).
#     Returns JSON containing prediction stats AND the base64-encoded Grad-CAM image.
#     """
#     model = app.state.model
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model failed to load on startup.")

#     # Read bytes
#     contents = await file.read()

#     try:
#         # Get processed input for model, and raw background for overlay
#         input_tensor, rgb_overlay_base = preprocess_for_gradcam(contents)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

#     # 1. Run inference
#     pred_prob = float(model.predict(input_tensor, verbose=0)[0][0])
#     has_fracture = bool(pred_prob >= 0.5)

#     # 2. Generate Grad-CAM visualization
#     try:
#         # A. Generate the raw heatmap (numpy array)
#         heatmap = generate_gradcam_heatmap(model, input_tensor, LAST_CONV_LAYER_NAME)

#         # B. Superimpose heatmap onto the original RGB-base image
#         # Assuming overlay_gradcam returns (superimposed_img_rgb, raw_heatmap_jet)
#         gradcam_rgb, _ = overlay_gradcam(rgb_overlay_base, heatmap, alpha=0.4)
#     except Exception as e:
#         # Fallback if Grad-CAM fails, don't crash the prediction
#         print(f"Warning: Grad-CAM failed: {e}")
#         gradcam_rgb = rgb_overlay_base # Or None

#     # 3. Encode the resultant RGB Grad-CAM image to Base64 string
#     base64_str = ""
#     if gradcam_rgb is not None:
#         # Convert RGB to BGR for OpenCV encoding to PNG/JPG
#         gradcam_bgr = cv2.cvtColor(gradcam_rgb, cv2.COLOR_RGB2BGR)

#         # Encode to PNG bytes in memory
#         _, buffer = cv2.imencode('.png', gradcam_bgr)

#         # Convert bytes to Base64 bytes -> to standard UTF-8 string
#         base64_str = base64.b64encode(buffer).decode('utf-8')

#     # 4. Return combined JSON response
#     return {
#         "filename": file.filename,
#         "is_fracture": has_fracture,
#         "prediction": "Fracture Detected" if has_fracture else "Normal",
#         "fracture_probability": round(pred_prob, 4),
#         "grad_cam_layer_targeted": LAST_CONV_LAYER_NAME,
#         # Frontend can use this directly: <img src="data:image/png;base64,{{grad_cam_base64}}" />
#         "grad_cam_base64": base64_str
#     }

@app.get("/check")
def root(): return {"status": "ok"}
