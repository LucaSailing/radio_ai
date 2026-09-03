import base64
from contextlib import asynccontextmanager
import io
import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf

# Import model loaders
from radio_ai_package.ml_logic.models.model_CNN import load_model_from_bucket
from radio_ai_package.ml_logic.models.model_vgg import load_vgg_model_from_bucket
from radio_ai_package.ml_logic.models.model_yolo import load_yolo_model_from_bucket

TARGET_SIZE = (224, 224)


# --- DISPATCHER HELPERS ---

def load_requested_model(model_choice: str):
    """Dispatcher to route model loading requests to the correct bucket loader."""
    choice = model_choice.lower().strip()
    if choice == "vgg":
        return load_vgg_model_from_bucket()
    elif choice == "yolo":
        return load_yolo_model_from_bucket()
    else:
        return load_model_from_bucket()  # Custom CNN default


# --- LIFESPAN MANAGER ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preloads all models into app.state at startup and manages shutdown cleanup."""
    print("Pre-loading deep learning models into app.state...")
    for choice in ["cnn", "vgg", "yolo"]:
        try:
            model = load_requested_model(choice)
            setattr(app.state, f"{choice}_model", model)
            print(f"✅ {choice.upper()} model pre-loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to pre-load {choice.upper()} model during startup: {e}")

    yield  # API serves requests here

    print("Shutting down API server... clearing loaded models.")


app = FastAPI(
    title="Radio AI API",
    version="1.0",
    description="Multi-task API for X-ray Classification (CNN/VGG + Grad-CAM) and Segmentation (YOLO)",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- GRAD-CAM & PREPROCESSING ENGINE ---

def preprocess_image_to_tensor(
    img_gray: np.ndarray,
    target_size: tuple[int, int] = TARGET_SIZE,
    num_channels: int = 1,
) -> tf.Tensor:
    """Converts a grayscale NumPy array into a normalized 4D float32 input tensor."""
    img_resized = cv2.resize(img_gray, target_size)
    img_normalized = img_resized.astype(np.float32) / 255.0

    if num_channels == 3:
        img_rgb = cv2.cvtColor(
            np.uint8(255 * img_normalized), cv2.COLOR_GRAY2RGB
        ).astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img_rgb, axis=0)
    else:
        input_tensor = np.expand_dims(
            np.expand_dims(img_normalized, axis=-1), axis=0
        )

    return tf.convert_to_tensor(input_tensor, dtype=tf.float32)


def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Recursively searches a Keras architecture for the last Conv2D layer."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name

    raise ValueError("No Conv2D layer found in the model architecture.")


def generate_gradcam_heatmap(
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
    last_conv_layer_name: str | None = None,
    target_mode: str = "fracture_only",
) -> np.ndarray:
    """Computes a robust Grad-CAM heatmap supporting sequential, functional, and nested models."""
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    target_conv_layer = None
    try:
        target_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # Resolve inner backbones (e.g., nested VGG)
        for layer in model.layers:
            if hasattr(layer, "layers"):
                try:
                    target_conv_layer = layer.get_layer(last_conv_layer_name)
                    break
                except ValueError:
                    continue

    if target_conv_layer is None:
        raise ValueError(f"Could not locate layer '{last_conv_layer_name}' in model.")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)

        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        pred_score = predictions[:, 0]
        loss = pred_score if target_mode == "fracture_only" else tf.where(pred_score >= 0.5, pred_score, 1.0 - pred_score)

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients could not be computed for requested layer.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)

    if max_val > 1e-8:
        heatmap = heatmap / max_val
    else:
        heatmap = tf.zeros_like(heatmap)

    return heatmap.numpy()


def overlay_gradcam(
    img_base_gray: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> tuple[np.ndarray, np.ndarray]:
    """Superimposes JET heatmap over the base image."""
    img_norm = (img_base_gray - img_base_gray.min()) / (
        img_base_gray.max() - img_base_gray.min() + 1e-8
    )
    img_uint8 = np.uint8(255 * img_norm)
    img_resized = cv2.resize(img_uint8, TARGET_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)

    heatmap_resized = cv2.resize(heatmap, TARGET_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_jet_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_jet_rgb = cv2.cvtColor(heatmap_jet_bgr, cv2.COLOR_BGR2RGB)

    superimposed_rgb = cv2.addWeighted(img_rgb, 1 - alpha, heatmap_jet_rgb, alpha, 0)
    return superimposed_rgb, heatmap_jet_rgb


def encode_image_to_base64(img_rgb: np.ndarray) -> str:
    """Encodes an RGB NumPy matrix into a Base64 PNG string."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


# --- API ENDPOINTS ---

@app.get("/")
def root():
    return {"status": "ok", "message": "Radio AI API is running"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    gt_file: UploadFile = File(None),  # Retained optional ground truth parameter
    task_type: str = Form("classification"),
    model_choice: str = Form("cnn"),
    target_mode: str = Form("fracture_only"),
):
    """Primary analysis endpoint routing classification (CNN/VGG) and segmentation (YOLO)."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Fetch Model from app.state or Lazy Load as Fallback
    model_choice_clean = model_choice.lower().strip()
    attr_name = f"{model_choice_clean}_model"
    model = getattr(app.state, attr_name, None)

    if model is None:
        try:
            model = load_requested_model(model_choice_clean)
            setattr(app.state, attr_name, model)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load requested model '{model_choice_clean}': {e}"
            )

    # 2. Process according to task type
    if task_type.lower() == "classification":
        if model_choice_clean not in ["cnn", "vgg"]:
            raise HTTPException(
                status_code=400, detail="Classification requires model_choice 'cnn' or 'vgg'."
            )

        np_arr = np.frombuffer(contents, np.uint8)
        img_gray = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        # Dynamically inspect expected channels (1 for CNN, 3 for VGG)
        model_channels = model.input_shape[-1] if model.input_shape[-1] else 1
        input_tensor = preprocess_image_to_tensor(
            img_gray, TARGET_SIZE, num_channels=model_channels
        )

        preds = model.predict(input_tensor, verbose=0)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        pred_prob = float(preds[0][0])
        is_fracture = bool(pred_prob >= 0.5)

        # Grad-CAM Visual Overlay
        gradcam_base64 = ""
        gradcam_layer = ""
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

            overlay_rgb, _ = overlay_gradcam(img_gray, heatmap, alpha=0.4)
            gradcam_base64 = encode_image_to_base64(overlay_rgb)
        except Exception as e:
            import traceback
            print(f"Grad-CAM Error: {e}\n{traceback.format_exc()}")
            gradcam_base64 = f"ERROR: {str(e)}"

        return {
            "task_type": "classification",
            "model_used": model_choice_clean,
            "filename": file.filename,
            "is_fracture": is_fracture,
            "prediction_label": "Fracture" if is_fracture else "Normal",
            "fracture_probability": round(pred_prob, 4),
            "target_mode_used": target_mode,
            "gradcam_layer": gradcam_layer,
            "gradcam_base64": gradcam_base64,
        }

    elif task_type.lower() == "segmentation":
        if model_choice_clean != "yolo":
            raise HTTPException(
                status_code=400, detail="Segmentation requires model_choice 'yolo'."
            )

        np_arr = np.frombuffer(contents, np.uint8)
        raw_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if raw_img is None:
            raise HTTPException(status_code=400, detail="Invalid image file format.")

        results = model(raw_img)[0]

        boxes = []
        if hasattr(results, "boxes") and results.boxes is not None:
            for box in results.boxes:
                coords = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                boxes.append({
                    "box": [round(c, 2) for c in coords],
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                })

        annotated_bgr = cv2.resize(results.plot(), TARGET_SIZE)
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
            status_code=400, detail="Invalid task_type. Choose 'classification' or 'segmentation'."
        )
