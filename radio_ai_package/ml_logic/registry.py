import os
import shutil
from datetime import datetime
import tensorflow as tf

from radio_ai_package.params import RAW_DATA_DIR


def save_model(model, model_name: str, model_type: str = "keras") -> str:
    """Sauvegarde un modèle en local, sous raw_data/models/, avec un nom versionné.

    model_type:
    - "keras" : CNN, VGG16 (tf.keras.Model) -> sauvegardé en .keras
    - "yolo"  : Ultralytics YOLO -> le meilleur checkpoint (best.pt) généré
                automatiquement pendant model.train() est copié vers le registre
    """
    models_dir = RAW_DATA_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if model_type == "keras":
        filepath = models_dir / f"{model_name}_fracture_{timestamp}.keras"
        model.save(filepath)

    elif model_type == "yolo":
        filepath = models_dir / f"{model_name}_fracture_{timestamp}.pt"
        best_ckpt = model.trainer.best  # chemin du meilleur poids après .train()
        shutil.copy(best_ckpt, filepath)

    else:
        raise ValueError(f"model_type inconnu : '{model_type}' (attendu : 'keras' ou 'yolo')")

    print(f"✅ Model saved at: {filepath}")
    return str(filepath)


def load_model(filepath: str, model_type: str = "keras"):
    """Charge un modèle depuis un chemin local.

    model_type:
    - "keras" : CNN, VGG16 -> tf.keras.models.load_model()
    - "yolo"  : Ultralytics YOLO -> ultralytics.YOLO(filepath)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model found at path: '{filepath}'")

    print(f"Loading model from: {filepath} ...")

    try:
        if model_type == "keras":
            model = tf.keras.models.load_model(filepath)
        elif model_type == "yolo":
            from ultralytics import YOLO  # import local pour flexibilité
            model = YOLO(filepath)
        else:
            raise ValueError(f"model_type inconnu : '{model_type}' (attendu : 'keras' ou 'yolo')")

        print("✅ Model loaded successfully!")
        return model

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
