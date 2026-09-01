import base64
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf

# Import deep learning model loader functions from custom package
from radio_ai_package.ml_logic.models.model_CNN import load_model_from_bucket
#from radio_ai_package.ml_logic.models.model_VGG import load_vgg_model
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
        MODELS["vgg"] = load_vgg_model()
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

    # Expand dimensions to create 4D batch shape (1, 224, 224, 1)
    input_tensor = np.expand_dims(np.expand_dims(img_normalized, axis=-1), axis=0)
    return tf.convert_to_tensor(input_tensor, dtype=tf.float32)


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Recursively searches a Keras model architecture and returns the name

    of the very last Conv2D layer found.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name

    raise ValueError("No Conv2D layer found in the provided model architecture.")


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
    last_conv_layer_name: str | None = None,
    target_mode: str = "fracture_only",
) -> np.ndarray:
    """Computes Grad-CAM heatmap for a binary classification model (Sigmoid activation).

    Accepts 4D input tensor shape: (1, H, W, C).
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Build sub-model mapping input tensor to [last conv output, final prediction score]
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output,
        ],
    )

    # Compute target loss and calculate gradients w.r.t. feature maps
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        pred_score = predictions[:, 0]

        if target_mode == "winning_class":
            loss = tf.where(pred_score >= 0.5, pred_score, 1.0 - pred_score)
        else:
            loss = pred_score

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU activation and normalize matrix between [0, 1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)

    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


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


# --- API ENDPOINTS ---


@app.post("/analyze")
async def analyze_xray(
    file: UploadFile = File(...),
    task_type: str = Form(..., description="'classification' or 'segmentation'"),
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

        # Preprocess grayscale image array to 4D tensor
        input_tensor = preprocess_image_to_tensor(img_gray, TARGET_SIZE)

        # Run binary classification inference
        pred_prob = float(model.predict(input_tensor, verbose=0)[0][0])
        has_fracture = bool(pred_prob >= 0.5)

        # Dynamically locate target layer and generate Grad-CAM visualization
        gradcam_base64 = ""
        last_conv_layer = ""
        try:
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

        # Run object detection inference
        results = yolo_model(raw_img)[0]

        # Format detection bounding boxes
        boxes = []
        for box in results.boxes:
            coords = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            boxes.append(
                {
                    "box": [round(c, 2) for c in coords],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                }
            )

        # Plot bounding boxes onto image and encode back to Base64
        annotated_bgr = results.plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        segmented_base64 = encode_image_to_base64(annotated_rgb)

        return {
            "task_type": "segmentation",
            "model_used": "yolo",
            "filename": file.filename,
            "detections_count": len(boxes),
            "detected_boxes": boxes,
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
