"""
preprocessor_vgg.py — préparation des données pour le modèle VGG16 (radio_ai)

Réutilise le pipeline CNN existant (filtering + preprocessing + make_dataset)
car VGG16 travaille sur les mêmes images en niveaux de gris que le CNN.
"""
from radio_ai_package.params import RAW_DATA_DIR
from radio_ai_package.ml_logic.preprocessors.preprocessor_CNN import filtering, preprocessing


def build_vgg_dataset(df):
    """Construit file_path (chemin ABSOLU, toujours via RAW_DATA_DIR), filtre le
    dataframe, puis split par patient + construit les tf.data.Dataset.
    Retourne (train_ds, val_ds, test_ds), (data_train, data_val, data_test)."""
    df = df.copy()
    df["file_path"] = df["filestem"].apply(
        lambda x: str(RAW_DATA_DIR / "cnn_images" / f"{x}.png")
    )
    df_filtered_CNN = filtering(df)
    return preprocessing(df_filtered_CNN)
