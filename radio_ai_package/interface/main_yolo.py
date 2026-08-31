"""
main_yolo.py — pipeline de détection de fractures par YOLO (radio_ai)
Miroir de main.py (pipeline CNN) : orchestre chargement des données,
préparation du dataset YOLO, entraînement et évaluation.
"""

from radio_ai_package.ml_logic.data import load_data  # adapte le nom si ta fonction s'appelle autrement (ex: load_df_with_local_paths)
from radio_ai_package.ml_logic.preprocessors.preprocessor_yolo import build_yolo_dataset
from radio_ai_package.ml_logic.models.model_yolo import (
    initialize_model_yolo,
    train_model_yolo,
    evaluate_model_yolo,
)


def train_yolo():
    """Charge les données, prépare le dataset YOLO, entraîne et évalue le modèle."""
    df = load_data()

    (yaml_path, dataset_dir), (data_train, data_val, data_test) = build_yolo_dataset(df)

    model = initialize_model_yolo()
    model, results = train_model_yolo(model, yaml_path)

    test_img_dir = dataset_dir / "images" / "test"
    test_lbl_dir = dataset_dir / "labels" / "test"
    metrics, df_eval = evaluate_model_yolo(model, test_img_dir, test_lbl_dir)

    return model, metrics


if __name__ == "__main__":
    train_yolo()
