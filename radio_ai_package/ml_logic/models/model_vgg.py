"""
model_vgg.py — initialisation, compilation, entraînement et évaluation du modèle VGG16 (radio_ai)
"""
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from radio_ai_package.params import IMG_SIZE, PATIENCE, LEARNING_RATE


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
