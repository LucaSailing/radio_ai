"""
model_vgg.py — initialisation, compilation, entraînement et évaluation du modèle VGG16 (radio_ai)
"""
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from radio_ai_package.params import IMG_SIZE, PATIENCE, LEARNING_RATE
from pathlib import Path
from tensorflow.keras.models import load_model
from google.cloud import storage

from radio_ai_package.params import BUCKET_NAME, MODEL_BUCKET_PREFIX_CNN


def initialize_model_vgg(img_size=IMG_SIZE) -> Model:
    """Construit le modèle VGG16 (transfer learning, base gelée) pour la
    classification fracture / pas fracture. img_size = tuple (H, W) comme dans params.py."""
    h, w = img_size

    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(h, w, 3))
    base_model.trainable = False  # on gèle les poids — on n'entraîne que la nouvelle tête

    inputs = layers.Input(shape=(h, w, 1))  # images en niveaux de gris
    x = layers.Concatenate()([inputs, inputs, inputs])  # duplique vers 3 canaux
    x = layers.Lambda(lambda img: preprocess_input(img * 255.0))(x)  # VGG16 attend [0,255] "caffe"
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs, name="vgg16_fracture")
    print("✅ Modèle VGG16 initialisé (base gelée, transfer learning)")
    return model


def compile_model_vgg(model: Model, learning_rate: float = LEARNING_RATE) -> Model:
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", "recall"],
    )
    print("✅ Modèle VGG16 compilé")
    return model


def train_model_vgg(model: Model, train_ds, val_ds, epochs: int = 20, patience: int = PATIENCE):
    es = EarlyStopping(patience=patience, restore_best_weights=True, monitor="val_loss")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[es])
    print(f"✅ Entraînement VGG16 terminé ({epochs} epochs max, patience={patience})")
    return model, history


def evaluate_model_vgg(model: Model, test_ds):
    metrics = model.evaluate(test_ds, return_dict=True)
    print(f"✅ Évaluation VGG16 — {metrics}")
    return metrics

def load_vgg_model_from_bucket(
    blob_name: str = None,
    local_dir: str = "/tmp/vgg_models",
) -> Model:
    """Downloads a saved VGG Keras model (.h5 or .keras) from GCS and loads it into memory.

    If `blob_name` is None, it automatically locates and loads the most recently updated
    model file inside the MODEL_BUCKET_PREFIX_CNN folder on GCS.

    Parameters:
    -----------
    blob_name : str, optional
        Specific GCS blob path (e.g., 'models/cnn/vgg16_fracture_20260831.keras').
    local_dir : str, optional
        Local cache directory where the downloaded file will be temporarily stored.

    Returns:
    --------
    Model : Loaded Keras VGG16 Model instance.
    """
    bucket = storage.Client().bucket(BUCKET_NAME)

    # 1. Fetch latest model automatically if no specific blob_name provided
    if not blob_name:
        blobs = list(bucket.list_blobs(prefix=f"{MODEL_BUCKET_PREFIX_CNN}/"))
        model_blobs = [
            b for b in blobs if b.name.endswith(".keras") or b.name.endswith(".h5")
        ]

        if not model_blobs:
            raise FileNotFoundError(
                f"No '.keras' or '.h5' model files found under gs://{BUCKET_NAME}/{MODEL_BUCKET_PREFIX_CNN}/"
            )

        # Sort by updated timestamp (newest first)
        model_blobs.sort(key=lambda b: b.updated, reverse=True)
        target_blob = model_blobs[0]
        print(f"🔍 Found latest VGG model on GCS: gs://{BUCKET_NAME}/{target_blob.name}")
    else:
        target_blob = bucket.blob(blob_name)
        if not target_blob.exists():
            raise FileNotFoundError(
                f"Model blob 'gs://{BUCKET_NAME}/{blob_name}' does not exist."
            )

    # 2. Prepare local cache path
    local_path = Path(local_dir) / Path(target_blob.name).name
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Download weights from GCS if not cached locally
    if not local_path.exists():
        print(f"⬇️ Downloading VGG model from GCS to local path: {local_path}...")
        target_blob.download_to_filename(str(local_path))
        print("✅ Download completed.")
    else:
        print(f"⚡ Using locally cached VGG model at: {local_path}")

    # 4. Load Keras model
    model = load_model(str(local_path))
    print("✅ VGG16 model loaded successfully into memory.")
    return model
