"""
main_vgg.py — pipeline de classification de fractures par VGG16 (transfer learning) (radio_ai)
"""
from radio_ai_package.ml_logic.data import load_data  # nom à vérifier/adapter selon data.py
from radio_ai_package.ml_logic.preprocessors.preprocessor_vgg import build_vgg_dataset
from radio_ai_package.ml_logic.models.model_vgg import (
    initialize_model_vgg, compile_model_vgg, train_model_vgg, evaluate_model_vgg,
)


def train_vgg():
    df = load_data()
    (train_ds, val_ds, test_ds), (data_train, data_val, data_test) = build_vgg_dataset(df)

    model = initialize_model_vgg()
    model = compile_model_vgg(model)
    model, history = train_model_vgg(model, train_ds, val_ds)
    metrics = evaluate_model_vgg(model, test_ds)

    return model, metrics


if __name__ == "__main__":
    train_vgg()
