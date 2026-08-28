import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from radio_ai_package.ml_logic.performance_metrics import get_x_test, get_y_test, get_binary_predictions

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


######################## Generating the heatmap ################################
def generate_gradcam_heatmap(model, img_array, last_conv_layer_name):
    """
    Computes standard Grad-CAM heatmap for a binary classification model (Sigmoid).
    """
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
        conv_outputs, predictions = grad_model(img_array)
        # Because we only have fracture or not and so
        # we are using a sigmoid activation,
        # we can Directly target index 0 (the positive class probability: the final guess)
        class_channel = predictions[:, 0]

    # 3. Compute gradients of class 1 probability (fracture) w.r.t. last conv output
    # If i changed the picture just a tiny bit, how would this affect your guess?
    # It calculates a gradient for each tiny piece of the image
    grads = tape.gradient(class_channel, conv_outputs)

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
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


####################### Generating the GC overlay ##############################
def overlay_gradcam(img_2d, heatmap, alpha=0.4):
    """
    Resizes heatmap to match image dimensions and overlays color map onto grayscale image.
    """
    # Ensure image is 2D uint8 scaled 0-255 (as opposed 0-1)
    if img_2d.max() <= 1.0:
        img_2d = (img_2d * 255).astype(np.uint8)
    else:
        img_2d = img_2d.astype(np.uint8)

    # Convert single-channel image to RGB for color overlay
    if len(img_2d.shape) == 2 or img_2d.shape[-1] == 1:
        img_rgb = cv2.cvtColor(img_2d.squeeze(), cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_2d.copy()

    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply Jet color map (Red = high activation, Blue = low activation)
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Superimpose color map onto original image
    superimposed_img = cv2.addWeighted(img_rgb, 1 - alpha, color_map, alpha, 0)
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
