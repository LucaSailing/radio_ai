import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import tensorflow as tf

from radio_ai_package.ml_logic.performance_metrics import get_x_test, get_y_test, get_binary_predictions, get_confusion_matrix_metrics, get_confusion_matrix_indices

####################### Getting the image input ################################

def get_image_sample(X_test, y_test=None, index=0):
    """Extracts a single image sample by index and prepares it for model inference.

    Returns:
        input_tensor: 4D numpy array with batch dimension (1, H, W, C)
        raw_image: 2D or 3D numpy array for plotting (H, W) or (H, W, C)
        label: ground truth label if y_test is provided, else None
    """
    raw_image = X_test[index]

    # Expand batch dimension for model input: (H, W, C) -> (1, H, W, C)
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
    """Converts a raw NumPy image array into a normalized 4D float32 input tensor.

    Shape returned: (1, height, width, num_channels)
    """
    # 1. Resize image to model expected dimensions
    img_resized = cv2.resize(img_array, target_size)

    # 2. Normalize pixel values to [0.0, 1.0]
    img_normalized = img_resized.astype(np.float32) / 255.0

    # 3. Add channel dimension if single-channel grayscale (H, W) -> (H, W, 1)
    if num_channels == 1 and len(img_normalized.shape) == 2:
        img_normalized = np.expand_dims(img_normalized, axis=-1)

    # 4. Add batch dimension (H, W, C) -> (1, H, W, C)
    input_tensor = np.expand_dims(img_normalized, axis=0)

    return tf.convert_to_tensor(input_tensor, dtype=tf.float32)






######################## Generating the heatmap ################################
def generate_gradcam_heatmap(
    model: tf.keras.Model,
    input_tensor: tf.Tensor,
    last_conv_layer_name: str = None,
    target_mode: str = "fracture_only") -> np.ndarray:
    """Computes standard Grad-CAM heatmap for a binary classification model (Sigmoid).
    Accepts a 4D input tensor shape: (1, H, W, C)."""
    # 0. Automatically locate the last Conv2D layer if not provided
    if last_conv_layer_name is None:
        last_conv_layer_name = [
            layer.name
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ][-1]

    # 1. Create a sub-model mapping input -> [target conv output, final model output]
    # This sub-model lets us look at two things at once:
    # What the detective's eyes saw in the middle layer.
    # The final answer at the end.
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,  # or model.layers[0].input
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # 2. Record operations for automatic differentiation
    # GradientTape works as a videocamera
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor)
        # Because we only have fracture or not and so
        # we are using a sigmoid activation,
        # we can Directly target index 0 (the positive class probability: the final guess)
        pred_score = predictions[:, 0]

        # Target class selection based on toggle mode
        if target_mode == "winning_class":
            loss = tf.where(
                pred_score >= 0.5, pred_score, 1.0 - pred_score
            )
        else:
            loss = pred_score

    # 3. Compute gradients of target loss w.r.t. last conv output
    # If i changed the picture just a tiny bit, how would this affect your guess?
    # It calculates a gradient for each tiny piece of the image
    grads = tape.gradient(loss, conv_outputs)

    # 4. Global average pooling: the weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Weight feature maps: we take everything that the model saw and multiply it by
    # how important each feature was: creates a perliminary heat map instead of having
    # a bunch of different ones (one per feature)
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    # @ matrix multiplication # tf.newaxis reshapes the pooled weights
    heatmap = tf.squeeze(heatmap)

    # 6. Apply ReLU and normalize: get rid of all negative contributors
    # We shrink or stretch all numbers so they are easily measured between
    # 0 (no interest) and 1 (super hot spot!).
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)

    # Return as a clean 2D NumPy float32 array
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


####################### Generating the GC overlay ##############################
def overlay_gradcam(img_2d, heatmap, alpha=0.4):
    """Resizes heatmap to match image dimensions and overlays color map onto grayscale image."""

    # 1. Safely normalize base image to [0, 1] and convert to 8-bit RGB
    img_gray = img_2d.squeeze()
    img_norm = (img_gray - img_gray.min()) / (
        img_gray.max() - img_gray.min() + 1e-8
    )
    img_uint8 = np.uint8(255 * img_norm)

    if len(img_uint8.shape) == 2:
        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_uint8.copy()

    # 2. Safely normalize heatmap to [0, 1] before scaling
    heatmap_norm = (heatmap - heatmap.min()) / (
        heatmap.max() - heatmap.min() + 1e-8
    )

    # 3. Resize heatmap to match base image dimensions
    heatmap_resized = cv2.resize(
        heatmap_norm, (img_rgb.shape[1], img_rgb.shape[0])
    )
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # 4. Apply Jet color map & convert OpenCV's BGR output to RGB:
    # (Red = high activation, Blue = low activation)
    color_map_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_map_rgb = cv2.cvtColor(color_map_bgr, cv2.COLOR_BGR2RGB)

    # 5. Superimpose color map onto original image
    superimposed_img = cv2.addWeighted(
        img_rgb, 1 - alpha, color_map_rgb, alpha, 0
    )

    return superimposed_img, heatmap_resized


###################### Looking at original and weights #########################
def plot_gradcam_comparison(
    model,
    image,
    true_label=None,
    last_conv_layer_name=None,
    alpha=0.4,
    figsize=(10, 5),
):
    """Generates and plots a side-by-side comparison of an original X-ray image
    and its Grad-CAM activation heatmap."""
    # 1. Automatically locate the last Conv2D layer if not provided
    if last_conv_layer_name is None:
        last_conv_layer_name = [
            layer.name
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ][-1]

    # 2. Ensure batch dimension is present (1, H, W, C)
    if len(image.shape) == 3:
        input_image = np.expand_dims(image, axis=0)
    else:
        input_image = image

    # 3. Predict probability and compute heatmap
    pred_prob = model.predict(input_image, verbose=0)[0][0]
    heatmap = generate_gradcam_heatmap(
        model, input_image, last_conv_layer_name
    )

    # 4. Create superimposed overlay image
    overlay_img, _ = overlay_gradcam(image.squeeze(), heatmap, alpha=alpha)

    # 5. Render side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Original Image
    axes[0].imshow(image.squeeze(), cmap="gray")
    title_str = "Original X-Ray"
    if true_label is not None:
        title_str += f" (True Label: {true_label})"
    axes[0].set_title(title_str)
    axes[0].axis("off")

    # Heatmap Overlay
    axes[1].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Grad-CAM Heatmap (Pred Prob: {pred_prob:.2f})")
    axes[1].axis("off")

    plt.tight_layout()
    return fig, axes

############################ Grad_cam images of TP, TN, FP and FN #########################################
def plot_gradcam_confusion_matrix(
    model,
    X_test,
    y_test,
    preds,
    binary_preds,
    last_conv_layer_name,
    alpha=0.4,
    viz_dir=Path("visualizations"),
    filename="gradcam_confusion_matrix.png",
):
    """Generates a 4x4 diagnostic grid of Grad-CAM heatmaps for TP, TN, FP, and FN categories.

    Accepts pre-computed NumPy arrays for fast execution without redundant model inference.
    """
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 0. Automatically locate the last Conv2D layer if not provided
    if last_conv_layer_name is None:
        last_conv_layer_name = [
            layer.name
            for layer in model.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ][-1]

    # 1. Ensure inputs are flat 1D NumPy arrays
    y_true = np.asarray(y_test).ravel().astype(int)
    y_probs = np.asarray(preds).ravel().astype(float)
    y_preds = np.asarray(binary_preds).ravel().astype(int)

     # 2. Extract array indices for each confusion matrix quadrant
    tp_indices, tn_indices, fp_indices, fn_indices = get_confusion_matrix_indices(
    y_test, binary_preds)

    ## Create a tuple of the different categories containing the pertinent
    ## indexes and a color code:
    categories = [
        ("True Positives (TP)", tp_indices, "green"),
        ("True Negatives (TN)", tn_indices, "blue"),
        ("False Positives (FP)", fp_indices, "orange"),
        ("False Negatives (FN)", fn_indices, "red"),
    ]

    # 3. Initialize 4x4 subplot grid
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(
        "GRAZPEDWRI-DX — Grad-CAM Confusion Matrix Grid",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )

# 4. Iterate over rows (categories) and columns (4 samples per category)
    for row_idx, (cat_name, indices, color) in enumerate(categories):
        # Sample up to 4 indices (randomized or top sequential)
        # row_idx  -> Determines the subplot row (0 to 3)
        # cat_name -> Used for row headers ("True Positives", etc.)
        # indices  -> The pool of images from which 4 random samples are picked
        # color    -> Applied to subplot borders and title colors
        selected_indices = (
            np.random.choice(indices, size=4, replace=False)
            if len(indices) >= 4
            else indices
        )

        for col_idx in range(4):
            ax = axes[row_idx, col_idx]

            # Handle case where a category has fewer than 4 total samples
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

            # Extracting the data for each photo
            idx = selected_indices[col_idx]
            raw_img = X_test[idx]

            # Extract scalars safely to avoid string formatting TypeErrors
            true_lbl = int(y_true[idx].item() if hasattr(y_true[idx], "item") else y_true[idx])
            pred_prob = float(y_probs[idx].item() if hasattr(y_probs[idx], "item") else y_probs[idx])
            pred_lbl = y_preds[idx]

            # Format input tensor (1, H, W, C) so it can be fed into both
            # Keras model and OpenCV functions without shape mismatch errors
            # **** input_tensor: A 4D batch tensor of shape (1, H, W, C)
            # required by Keras models for inference.
            # **** img_gray: A 2D grayscale image array of shape (H, W)
            # required by OpenCV for Grad-CAM visualization blending.
            if len(raw_img.shape) == 2:
                # Condition: Triggered if the image is a plain 2D matrix
                # (shape (256, 256)):
                # **** input_tensor: np.expand_dims(..., axis=(0, -1))
                # adds a batch dimension at the front (axis 0)
                # and a channel dimension at the back (axis -1),
                # turning (256, 256) into (1, 256, 256, 1).
                # **** img_gray: Kept as raw_img directly since it is already 2D.
                input_tensor = np.expand_dims(raw_img, axis=(0, -1))
                img_gray = raw_img
            elif raw_img.shape[-1] == 1:
                # Condition: Triggered if the image has an explicit single-channel dimension
                # (shape (256, 256, 1)):
                # *** input_tensor: Adds only the batch dimension at axis 0,
                # resulting in (1, 256, 256, 1).
                # *** img_gray: .squeeze(-1) removes the trailing single-channel dimension,
                # reducing (256, 256, 1) to a 2D array (256, 256)
                input_tensor = np.expand_dims(raw_img, axis=0)
                img_gray = raw_img.squeeze(-1)
            else:
                # Condition: Triggered if the image has 3 color channels
                # (e.g., shape (256, 256, 3)):
                # *** input_tensor: Adds the batch dimension at axis 0,
                # resulting in (1, 256, 256, 3).
                # ***img_gray: Converts the 3-channel RGB image into a single
                # 2D grayscale matrix using cv2.cvtColor, ensuring the base
                # X-ray image renders as monochrome under the colored heatmaps.
                input_tensor = np.expand_dims(raw_img, axis=0)
                img_gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)

            # Generate Heatmap
            heatmap = generate_gradcam_heatmap(
                model, input_tensor, last_conv_layer_name)

             # Create Superimposed RGB Image
             ## Grad-CAM heatmaps are extracted from the last convolutional layer,
             ## which is much smaller than the original input image
             ## OpenCV's resize stretches the heatmap back up to match the exact
             # width and height of your input image (img_gray.shape[1] is width,
             # img_gray.shape[0] is height).
            heatmap_resized = cv2.resize(
                heatmap, (img_gray.shape[1], img_gray.shape[0]))

             ## Converting continuous floating-point values (0.0 to 1.0)
             ## into an 8-bit unsigned integer range (0 to 255), which is required by
             ## OpenCV's color mapping functions.
            heatmap_uint8 = np.uint8(255 * heatmap_resized)

            ## Applying the JET colormap, transforming grayscale intensity \
            ## values into a thermal color spectrum
            heatmap_color = cv2.applyColorMap(
                heatmap_uint8, cv2.COLORMAP_JET
            )

            ## OpenCV natively produces images in BGR (Blue-Green-Red) order.
            ## Matplotlib requires RGB (Red-Green-Blue), so this swaps the channels
            ## so colors display correctly on the figure.
            heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            ## Normalizes the 2D grayscale X-ray into an 8-bit image (0–255) and
            ## converts it to a 3-channel RGB representation so it can be
            ## mathematically blended with the 3-channel heatmap.
            img_rgb = cv2.cvtColor(
                np.uint8(img_gray * 255), cv2.COLOR_GRAY2RGB
            )
            ## Blends the two RGB images using a weighted linear combination
            ## With alpha=0.4, the final image is 60% original X-ray structural detail
            # and 40% transparent color heatmap overlay.
            overlay = cv2.addWeighted(
                img_rgb, 1 - alpha, heatmap_rgb, alpha, 0
            )


            # Render image on subplot
            ax.imshow(overlay)
            ax.axis("off")

            # Title & Metadata Annotation
            title_text = f"Prob: {pred_prob[0]:.3f} | True: {true_lbl}"
            ax.set_title(
                f"Idx: {idx} | Prob: {pred_prob[0]:.3f} | True: {true_lbl}",
                fontsize=10,
                fontweight="bold",
            )

            # Draw colored bounding box border per category
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)
                spine.set_visible(True)
        # Row Header Labels on the far left
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

    # Save output
    out_path = viz_dir / filename
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Grad-CAM Confusion Matrix saved to: {out_path}")
