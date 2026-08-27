# Cleaning and preprocessing

import tensorflow as tf
from radio_ai_package.params import (IMG_SIZE, BATCH_SIZE, TEST_SIZE, VAL_SIZE,
                                     RANDOM_STATE, PATH_COL)


def filtering(df):
    """Ne garde que les cas exploitables pour le CNN : sans plâtre, métal,
    incertitude de diagnostic ni ostéopénie, et avec projection < 3.
    Supprime ensuite les colonnes non utilisées par le modèle."""
    df_filtered_CNN = df[df['cast'].isna()
                         & df['metal'].isna()
                         & df['diagnosis_uncertain'].isna()
                         & df['osteopenia'].isna()
                         & (df['projection'] < 3)]

    df_filtered_CNN = df_filtered_CNN.drop(
        columns=['cast', 'metal', 'diagnosis_uncertain', 'device_manufacturer',
                 'timehash', 'ao_classification', 'initial_exam', 'study_number',
                 'osteopenia'], errors='ignore')

    return df_filtered_CNN


def load_image(path, label):
    """Lit un PNG en niveaux de gris, redimensionne en PRÉSERVANT le ratio
    (padding pour compléter au carré), normalise dans [0, 1]."""
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize_with_pad(image, IMG_SIZE[0], IMG_SIZE[1])  # ratio préservé
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(dataframe, training=False, path_col=PATH_COL):
    """Construit un tf.data.Dataset depuis un DataFrame. Shuffle uniquement à
    l'entraînement (inutile et non reproductible sur val/test)."""
    dataset = tf.data.Dataset.from_tensor_slices((
        dataframe[path_col].values,
        dataframe["fracture_visible"].values
    ))

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()               # <- télécharge/décode 1x, réutilise ensuite

    if training:
        dataset = dataset.shuffle(len(dataframe), reshuffle_each_iteration=True)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def preprocessing(df_filtered_CNN, path_col=PATH_COL):
    """Split PAR PATIENT. path_col = colonne de chemins à utiliser
    ('file_path' en local, 'file_path_gs' en distant, dérivé de DATA_MODE)."""
    data_model = df_filtered_CNN.copy()
    data_model['fracture_visible'] = data_model['fracture_visible'].fillna(0)
    data_model = data_model[data_model[path_col] != ""]      # retire les chemins vides

    patients = data_model['patient_id'].drop_duplicates().sample(
        frac=1, random_state=RANDOM_STATE).tolist()

    n = len(patients)
    i_train = round(n * (1 - TEST_SIZE - VAL_SIZE))
    i_val   = round(n * (1 - TEST_SIZE))

    train_ids = set(patients[:i_train])
    val_ids   = set(patients[i_train:i_val])
    test_ids  = set(patients[i_val:])

    data_train = data_model[data_model['patient_id'].isin(train_ids)]
    data_val   = data_model[data_model['patient_id'].isin(val_ids)]
    data_test  = data_model[data_model['patient_id'].isin(test_ids)]

    train_ds = make_dataset(data_train, training=True, path_col=path_col)
    val_ds   = make_dataset(data_val,   path_col=path_col)
    test_ds  = make_dataset(data_test,  path_col=path_col)

    return (train_ds, val_ds, test_ds), (data_train, data_val, data_test)
