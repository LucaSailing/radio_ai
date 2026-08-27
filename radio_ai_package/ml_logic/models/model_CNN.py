# ---------- modèle : init, compile, train, evaluate, save ----------

import time
import tensorflow as tf
from tensorflow.keras import Sequential, layers, callbacks, optimizers
from tensorflow.keras.layers import Input
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from datetime import datetime

from radio_ai_package.params import (IMG_SIZE, EPOCHS, PATIENCE, LEARNING_RATE,
                                     BUCKET_NAME, MODEL_BUCKET_PREFIX)


def initialize_model():
    """CNN pour classification binaire de radios en niveaux de gris.
    Entrée 528x528x1, sortie sigmoïde (proba de fracture)."""
    model = Sequential()
    model.add(Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)))
    model.add(layers.Conv2D(16, (3, 3), padding='same', activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(32, (2, 2), padding='same', activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))       # couche intermédiaire
    model.add(layers.Dense(1, activation='sigmoid'))     # proba de fracture
    return model

# def initialize_model():
#     """CNN pour classification binaire de radios en niveaux de gris.
#     4 blocs conv + GlobalAveragePooling (au lieu de Flatten) + dropout.
#     Lit IMG_SIZE depuis params.py -> reste synchronisé avec le pipeline."""
#     model = Sequential([
#         layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),   # <- IMG_SIZE, pas 256 en dur

#         # Bloc 1
#         layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),

#         # Bloc 2
#         layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),

#         # Bloc 3
#         layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),

#         # Bloc 4
#         layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),

#         layers.GlobalAveragePooling2D(),        # remplace Flatten
#         layers.Dense(50, activation="relu"),
#         layers.Dropout(0.5),
#         layers.Dense(1, activation="sigmoid"),  # proba de fracture
#     ])
#     return model

def compile_model(model):
    """Compile pour classification binaire. Au-delà de l'accuracy (trompeuse
    sur un jeu déséquilibré), on suit precision, recall et AUC — les métriques
    qui comptent vraiment pour du dépistage médical."""
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
        ]
    )
    return model


def compute_class_weights(y):
    """Poids de classe pour compenser le déséquilibre (fractures minoritaires).
    Calculé directement sur la série d'étiquettes du train — instantané, pas
    besoin de re-parcourir le tf.data.Dataset."""
    y = np.asarray(y).astype(int)
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}



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
        class_weight=class_weight,
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
    """Sauvegarde le modèle avec un horodatage, pour ne pas écraser les runs
    précédents et pouvoir comparer les versions."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = MODEL_DIR / f"{name}_{timestamp}.keras"
    model.save(path)
    print(f"Modèle sauvegardé : {path}")
    return path
