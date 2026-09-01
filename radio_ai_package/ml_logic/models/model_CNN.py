# ---------- modèle : init, compile, train, evaluate, save ----------

import time
import tempfile
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers, callbacks, optimizers
from tensorflow.keras.layers import Input
from sklearn.utils.class_weight import compute_class_weight
from google.cloud import storage

from radio_ai_package.params import (IMG_SIZE, EPOCHS, PATIENCE, LEARNING_RATE,
                                     BUCKET_NAME, MODEL_BUCKET_PREFIX)


# def initialize_model():
#     """CNN pour classification binaire de radios en niveaux de gris.
#     Entrée 528x528x1, sortie sigmoïde (proba de fracture)."""
#     model = Sequential()
#     model.add(Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)))
#     model.add(layers.Conv2D(16, (3, 3), padding='same', activation="relu"))
#     model.add(layers.MaxPool2D(pool_size=(2, 2)))
#     model.add(layers.Conv2D(32, (2, 2), padding='same', activation="relu"))
#     model.add(layers.MaxPool2D(pool_size=(2, 2)))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(50, activation='relu'))       # couche intermédiaire
#     model.add(layers.Dense(1, activation='sigmoid'))     # proba de fracture
#     return model

def initialize_model():

    def conv_block(filters):
        return [
            layers.Conv2D(
                filters,
                (3, 3),
                padding="same",
                use_bias=False
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPooling2D(pool_size=(2, 2))
        ]

    model = Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),

        # Data augmentation
        layers.RandomRotation(0.03),
        layers.RandomTranslation(0.03, 0.03),
        layers.RandomContrast(0.10),

        # CNN
        *conv_block(16),
        *conv_block(32),
        *conv_block(64),
        *conv_block(128),

        # Remplace Flatten
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(1, activation="sigmoid")
    ])

    return model


def compile_model(model):

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]
    )

    return model


def compute_class_weights(y):

    y = np.asarray(y).astype(int)

    classes = np.unique(y)

    weights = compute_class_weight(
        "balanced",
        classes=classes,
        y=y
    )

    return {
        int(c): float(w)
        for c, w in zip(classes, weights)
    }



def warmup_cache(dataset, nom="dataset"):
    """Force le remplissage du cache en itérant une fois sur le dataset,
    sans entraîner. Chronomètre le coût de chargement (téléchargement +
    décodage) — c'est LE chiffre qui diffère entre local et remote."""
    t0 = time.time()
    n_batches = 0
    n_images = 0
    for images, labels in dataset:
        n_batches += 1
        n_images += images.shape[0]
    dt = time.time() - t0
    print(f"  ⏱  cache [{nom}] rempli : {dt:.1f}s "
          f"({n_images} images, {n_batches} batches, {n_images/dt:.0f} img/s)")
    return dt


def train_model(model, ds_train, ds_val, y_train=None):
    """Entraîne avec EarlyStopping (monitore val_auc, restaure les meilleurs
    poids) et réduction du learning rate sur plateau. Si y_train est fourni,
    applique des poids de classe. Retourne (model, history)."""

    cbs = [
        callbacks.ModelCheckpoint(
            filepath=f"checkpoints/fracture_cnn_epoch{{epoch:02d}}.keras",

        ),
        callbacks.EarlyStopping(
            monitor='val_auc', mode='max',
            patience=PATIENCE, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=max(1, PATIENCE // 2), min_lr=1e-6, verbose=1),
    ]

    class_weight = None
    if y_train is not None:
        class_weight = compute_class_weights(y_train)
        print(f"  Poids de classe appliqués : {class_weight}")

    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        #class_weight=class_weight,
        callbacks=cbs,
        verbose=1
    )
    return model, history


def evaluate_model(model, ds_test):
    """Évalue sur le test (jamais vu à l'entraînement) et affiche les métriques."""
    results = model.evaluate(ds_test, verbose=1, return_dict=True)
    print("\nRésultats sur le test :")
    for name, value in results.items():
        print(f"  {name:12s} : {value:.4f}")
    return results

def save_model(model, name="cnn_fracture"):
    """Sauvegarde le modèle horodaté dans le bucket GCS (dossier models/),
    pas en local. Passe par un fichier temporaire car model.save() ne peut pas
    écrire directement dans GCS. Retourne l'URI gs://."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{name}_{timestamp}.keras"
    blob_name = f"{MODEL_BUCKET_PREFIX}/{filename}"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = f"{tmpdir}/{filename}"
        model.save(local_path)                       # écriture locale temporaire

        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        bucket.blob(blob_name).upload_from_filename(local_path)

    gs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
    print(f"  Modèle sauvegardé dans le bucket : {gs_uri}")
    return gs_uri


def _latest_model_blob(bucket):
    """Retourne le blob du modèle le plus récent dans models/ (par nom, qui
    contient l'horodatage -> tri alphabétique = tri chronologique)."""
    blobs = [b for b in bucket.list_blobs(prefix=f"{MODEL_BUCKET_PREFIX}/")
             if b.name.endswith(".keras")]
    if not blobs:
        raise FileNotFoundError(f"Aucun modèle .keras dans gs://{BUCKET_NAME}/{MODEL_BUCKET_PREFIX}/")
    return max(blobs, key=lambda b: b.name)   # nom horodaté -> le plus grand = le plus récent


def load_model_from_bucket(filename=None):
    """Charge un modèle depuis le bucket. Si filename est None, prend le plus
    récent. Télécharge dans un fichier temporaire puis keras.load_model.
    Retourne le modèle Keras prêt à évaluer."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    if filename:
        blob = bucket.blob(f"{MODEL_BUCKET_PREFIX}/{filename}")
        if not blob.exists():
            raise FileNotFoundError(f"Modèle introuvable : {filename}")
    else:
        blob = _latest_model_blob(bucket)

    print(f"  Chargement du modèle : gs://{BUCKET_NAME}/{blob.name}")
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = f"{tmpdir}/model.keras"
        blob.download_to_filename(local_path)
        model = keras.models.load_model(local_path)

    return model
