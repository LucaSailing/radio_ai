"""
model_vgg.py — initialisation, compilation, entraînement et évaluation du modèle VGG16 (radio_ai)
"""

import tempfile
from datetime import datetime

from google.cloud import storage
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from radio_ai_package.params import (
    BUCKET_NAME,
    IMG_SIZE,
    LEARNING_RATE,
    MODEL_BUCKET_PREFIX,
    PATIENCE,
)


def initialize_model_vgg(img_size=IMG_SIZE) -> Model:
    """Construit le modèle VGG16 (transfer learning, base gelée) pour la
    classification fracture / pas fracture. img_size = tuple (W, H) comme dans params.py."""
    w, h = img_size  # IMG_SIZE = (WIDTH, HEIGHT) dans params.py

    # Explicitement nommer base_model 'vgg16' pour get_layer()
    base_model = VGG16(
        weights="imagenet", include_top=False, input_shape=(h, w, 3), name="vgg16"
    )
    base_model.trainable = False  # on gèle les poids — on n'entraîne que la nouvelle tête

    inputs = layers.Input(shape=(h, w, 1))  # images en niveaux de gris
    x = layers.Concatenate()([inputs, inputs, inputs])  # duplique vers 3 canaux
    x = layers.Lambda(lambda img: preprocess_input(img * 255.0))(
        x
    )  # VGG16 attend [0,255] "caffe"
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


def train_model_vgg(
    model: Model, train_ds, val_ds, y_train=None, epochs: int = 20, patience: int = PATIENCE
):
    es = EarlyStopping(
        patience=patience, restore_best_weights=True, monitor="val_loss"
    )
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=[es]
    )
    print(f"✅ Entraînement VGG16 terminé ({epochs} epochs max, patience={patience})")
    return model, history


def evaluate_model_vgg(model: Model, test_ds):
    metrics = model.evaluate(test_ds, return_dict=True)
    print(f"✅ Évaluation VGG16 — {metrics}")
    return metrics


def fine_tune_model_vgg(
    model: Model,
    train_ds,
    val_ds,
    epochs: int = 15,
    patience: int = PATIENCE,
    fine_tune_at: str = "block5_conv1",
    fine_tune_lr: float = 1e-5,
):
    """Dégèle les couches de VGG16 à partir de `fine_tune_at` (inclus) et poursuit
    l'entraînement avec un learning rate très bas."""

    # Extraction sécurisée du sous-modèle VGG16
    try:
        base_model = model.get_layer("vgg16")
    except ValueError:
        # Fallback au cas où le nom aurait changé au moment de l'import/load
        base_model = [l for l in model.layers if "vgg16" in l.name.lower()][0]

    base_model.trainable = True

    set_trainable = False
    for layer in base_model.layers:
        if layer.name == fine_tune_at:
            set_trainable = True
        layer.trainable = set_trainable

    model.compile(
        optimizer=Adam(learning_rate=fine_tune_lr),
        loss="binary_crossentropy",
        metrics=["accuracy", "recall"],
    )
    print(f"✅ Fine-tuning activé à partir de '{fine_tune_at}' (lr={fine_tune_lr})")

    es = EarlyStopping(
        patience=patience, restore_best_weights=True, monitor="val_loss"
    )
    history_ft = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=[es]
    )
    print(f"✅ Fine-tuning terminé ({epochs} epochs max, patience={patience})")
    return model, history_ft


def save_model(model, name="vgg_fracture"):
    """Sauvegarde le modèle horodaté dans le bucket GCS (dossier models/)."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{name}_{timestamp}.keras"
    blob_name = f"{MODEL_BUCKET_PREFIX}/{filename}"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = f"{tmpdir}/{filename}"
        model.save(local_path)

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        bucket.blob(blob_name).upload_from_filename(local_path)

    gs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
    print(f"  Modèle sauvegardé dans le bucket : {gs_uri}")
    return gs_uri

def load_vgg_model_from_bucket(
    filename: str = "models/VGG/vgg16_fracture_20260902-145242.keras",
):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    blob = bucket.blob(filename)
    if not blob.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : gs://{BUCKET_NAME}/{filename}"
        )

    print(f"  Chargement du modèle : gs://{BUCKET_NAME}/{blob.name}")
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = f"{tmpdir}/model.keras"
        blob.download_to_filename(local_path)# Patch de la classe Lambda pour forcer la forme de sortie
        class FixedLambda(layers.Lambda):
            def __init__(self, *args, **kwargs):
                kwargs["output_shape"] = lambda input_shape: input_shape
                super().__init__(*args, **kwargs)

            def compute_output_shape(self, input_shape):
                return input_shape

        custom_objects = {
            "Lambda": FixedLambda,
            "<lambda>": lambda img: preprocess_input(img * 255.0),
            "preprocess_input": preprocess_input,
        }

        model = keras.models.load_model(local_path, custom_objects=custom_objects, compile=False, safe_mode=False)

    return model
