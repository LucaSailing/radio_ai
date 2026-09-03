from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from radio_ai_package.ml_logic.performance_metrics import (
    get_confusion_matrix_indices,
)

####################### Getting the image input ################################

def get_image_sample(X_test, y_test=None, index=0):
    """Extracts a single image sample by index and prepares it for model inference."""
    raw_image = X_test[index]

    if len(raw_image.shape) == 3:
        input_tensor = np.expand_dims(raw_image, axis=0)
    elif len(raw_image.shape) == 2:
        input_tensor = np.expand_dims(raw_image, axis=(0, -1))
    else:
        input_tensor = raw_image

    label = y_test[index] if y_test is not None else None
    return input_tensor, raw_image, label


################ Transforming the image array into a 4D tensor #################

def preprocess_image_to_tensor(
    img_array: np.ndarray,
    target_size: tuple = (224, 224),
    num_channels: int = 1,
) -> tf.Tensor:
    """Converts a raw NumPy image array into a normalized 4D float32 input tensor (1, H, W, C)."""
    img_resized = cv2.resize(img_array, target_size)
    img_normalized = img_resized.astype(np.float32)

    # Safe normalization check: scale to [0, 1] only if input is in [0, 255]
    if img_normalized.max() > 1.0:
        img_normalized = img_normalized / 255.0

    if num_channels == 1 and len(img_normalized.shape) == 2:
        img_normalized = np.expand_dims(img_normalized, axis=-1)
    elif num_channels == 1 and img_normalized.ndim == 3 and img_normalized.shape[-1] != 1:
        img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_RGB2GRAY)[..., np.newaxis]

    if img_normalized.ndim == 3:
        img_normalized = np.expand_dims(img_normalized, axis=0)

    return tf.convert_to_tensor(img_normalized, dtype=tf.float32)


###################### Getting the last convolutional layer ####################

def find_last_conv_layer(model: tf.keras.Model) -> str:
    """Recursively searches a Keras model architecture and returns the name of the last Conv2D layer found."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name

    raise ValueError("No Conv2D layer found in the provided model architecture.")


def find_vgg_base(model: tf.keras.Model) -> tf.keras.Model:
    """Extracts the inner VGG16 base sub-model from the outer model."""
    try:
        return model.get_layer("vgg16")
    except ValueError:
        for layer in model.layers:
            if "vgg16" in layer.name.lower() and hasattr(layer, "get_layer"):
                return layer
    raise ValueError("Could not locate nested VGG16 base model.")


######################## Generating the heatmap ################################

def generate_gradcam_heatmap(
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
    last_conv_layer_name: str | None = None,
    target_mode: str = "fracture_only",
) -> np.ndarray:
    """Computes Grad-CAM heatmap supporting flat custom CNNs and nested VGG16 architectures."""

    # 1. Check if model has a nested VGG base
    has_vgg = any("vgg16" in l.name.lower() for l in model.layers)

    if has_vgg:
        vgg_base = find_vgg_base(model)
        if last_conv_layer_name is None:
            last_conv_layer_name = "block5_conv3"

        # Construct intermediate model mapping VGG inputs -> Target Conv layer output
        target_conv = vgg_base.get_layer(last_conv_layer_name)
        conv_sub_model = tf.keras.Model(inputs=vgg_base.inputs, outputs=target_conv.output)

        with tf.GradientTape() as tape:
            x = input_tensor
            for layer in model.layers:
                if layer == vgg_base:
                    break
                x = layer(x)

            conv_outputs = conv_sub_model(x)
            tape.watch(conv_outputs)

            x_head = conv_outputs
            start_idx = [l.name for l in vgg_base.layers].index(last_conv_layer_name) + 1
            for layer in vgg_base.layers[start_idx:]:
                x_head = layer(x_head)

            vgg_idx = model.layers.index(vgg_base)
            for layer in model.layers[vgg_idx + 1:]:
                x_head = layer(x_head)

            predictions = x_head
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]

            pred_score = predictions[:, 0]

            if target_mode == "winning_class":
                loss = tf.where(pred_score >= 0.5, pred_score, 1.0 - pred_score)
            elif target_mode == "normal_only":
                loss = 1.0 - pred_score
            else:  # 'fracture_only' default
                loss = pred_score

    else:
        # Standard Flat Model Logic (Custom CNNs / Sequential)
        if last_conv_layer_name is None:
            last_conv_layer_name = "last_block_conv" if "last_block_conv" in [l.name for l in model.layers] else find_last_conv_layer(model)

        # Build symbolic feature extraction model
        target_conv_layer = model.get_layer(last_conv_layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[target_conv_layer.output, model.output],
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_tensor)

            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]

            pred_score = predictions[:, 0]

            if target_mode == "winning_class":
                loss = tf.where(pred_score >= 0.5, pred_score, 1.0 - pred_score)
            elif target_mode == "normal_only":
                loss = 1.0 - pred_score
            else:  # 'fracture_only' default
                loss = pred_score

    # Extract gradients & compute weighted heatmap
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)

    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


####################### Generating the GC overlay ##############################

def overlay_gradcam(img_2d: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4):
    """Resizes heatmap to match image dimensions and overlays color map onto grayscale image."""
    img_gray = np.asarray(img_2d)
    while img_gray.ndim > 2:
        img_gray = img_gray.squeeze()
        if img_gray.ndim == 3 and img_gray.shape[-1] == 1:
            img_gray = img_gray[..., 0]
        elif img_gray.ndim == 3 and img_gray.shape[-1] == 3:
            img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
            break

    # Safe 8-bit conversion
    if img_gray.max() <= 1.0:
        img_uint8 = np.uint8(255 * img_gray)
    else:
        img_uint8 = np.uint8(img_gray)

    img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    denom = (heatmap_max - heatmap_min) if (heatmap_max - heatmap_min) > 0 else 1e-8
    heatmap_norm = (heatmap - heatmap_min) / denom

    heatmap_resized = cv2.resize(
        heatmap_norm, (img_rgb.shape[1], img_rgb.shape[0])
    )
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    color_map_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_map_rgb = cv2.cvtColor(color_map_bgr, cv2.COLOR_BGR2RGB)

    superimposed_img = cv2.addWeighted(
        img_rgb, 1 - alpha, color_map_rgb, alpha, 0
    )

    return superimposed_img, heatmap_resized


###################### Looking at original and weights #########################

def plot_gradcam_comparison(
    model: tf.keras.Model,
    image: np.ndarray,
    true_label: int | None = None,
    last_conv_layer_name: str | None = None,
    alpha: float = 0.4,
    figsize: tuple = (10, 5),
):
    """Generates and plots a side-by-side comparison of an original X-ray image and its Grad-CAM overlay."""
    input_tensor = preprocess_image_to_tensor(image)

    preds = model.predict(input_tensor, verbose=0)
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    pred_prob = float(preds[0][0])

    heatmap = generate_gradcam_heatmap(
        model, input_tensor, last_conv_layer_name=last_conv_layer_name
    )

    overlay_img, _ = overlay_gradcam(image, heatmap, alpha=alpha)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(image.squeeze(), cmap="gray")
    title_str = "Original X-Ray"
    if true_label is not None:
        title_str += f" (True Label: {true_label})"
    axes[0].set_title(title_str)
    axes[0].axis("off")

    axes[1].imshow(overlay_img)
    axes[1].set_title(f"Grad-CAM Heatmap (Pred Prob: {pred_prob:.2f})")
    axes[1].axis("off")

    plt.tight_layout()
    return fig, axes


############################ Grad_cam images of TP, TN, FP and FN #########################################

def plot_gradcam_confusion_matrix(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    preds: np.ndarray,
    binary_preds: np.ndarray,
    last_conv_layer_name: str | None = None,
    alpha: float = 0.4,
    viz_dir: Path = Path("visualizations"),
    filename: str = "gradcam_confusion_matrix.png",
):
    """Generates a 4x4 diagnostic grid of Grad-CAM heatmaps for TP, TN, FP, and FN categories."""
    viz_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_test).ravel().astype(int)
    y_probs = np.asarray(preds).ravel().astype(float)

    tp_indices, tn_indices, fp_indices, fn_indices = get_confusion_matrix_indices(
        y_test, binary_preds
    )

    categories = [
        ("True Positives (TP)", tp_indices, "green"),
        ("True Negatives (TN)", tn_indices, "blue"),
        ("False Positives (FP)", fp_indices, "orange"),
        ("False Negatives (FN)", fn_indices, "red"),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(
        "GRAZPEDWRI-DX — Grad-CAM Confusion Matrix Grid",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

    for row_idx, (cat_name, indices, color) in enumerate(categories):
        selected_indices = (
            np.random.choice(indices, size=4, replace=False)
            if len(indices) >= 4
            else indices
        )

        for col_idx in range(4):
            ax = axes[row_idx, col_idx]

            if col_idx >= len(selected_indices):
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    "N/A (Insufficient Samples)",
                    ha="center",
                    va="center",
                )
                continue

            idx = selected_indices[col_idx]
            raw_img = X_test[idx]

            true_lbl = int(y_true[idx].item() if hasattr(y_true[idx], "item") else y_true[idx])
            pred_prob = float(y_probs[idx].item() if hasattr(y_probs[idx], "item") else y_probs[idx])

            input_tensor = preprocess_image_to_tensor(raw_img)

            # Generate heatmap & construct overlay
            heatmap = generate_gradcam_heatmap(
                model, input_tensor, last_conv_layer_name=last_conv_layer_name
            )
            overlay, _ = overlay_gradcam(raw_img, heatmap, alpha=alpha)

            ax.imshow(overlay)
            ax.axis("off")

            ax.set_title(
                f"Idx: {idx} | Prob: {pred_prob:.3f} | True: {true_lbl}",
                fontsize=10,
                fontweight="bold",
            )

            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)

        axes[row_idx, 0].text(
            -0.12,
            0.5,
            cat_name,
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            fontsize=13,
            fontweight="bold",
            va="center",
            ha="center",
            color=color,
        )

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.96])

    out_path = viz_dir / filename
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Grad-CAM Confusion Matrix saved to: {out_path}")
