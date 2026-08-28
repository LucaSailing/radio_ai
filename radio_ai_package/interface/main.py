# ============================================================================
#  main.py — pipeline détection de fractures (radio_ai)
# ============================================================================

import os

# doit être défini AVANT l'import de tensorflow / google
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from radio_ai_package.params import *

# --- data (Luca) ---
from radio_ai_package.ml_logic.data import load_data
from radio_ai_package.ml_logic.preprocessors.preprocessor_CNN import filtering, preprocessing
from radio_ai_package.ml_logic.viz import show_train_samples, show_predictions

# --- modèle (Modibo / Merwan) ---
from radio_ai_package.ml_logic.models.model_CNN import (initialize_model, compile_model,
                                                        train_model, evaluate_model,
                                                        save_model, warmup_cache,
                                                        load_model_from_bucket)

# --- métriques (Mariana) ---
from radio_ai_package.ml_logic.performance_metrics import (
    get_user_threshold, get_predictions, get_x_test, get_y_test, get_binary_predictions,
    get_confusion_matrix, confusion_matrix_display_predicted,
    comparing_metrics_predictions, get_classification_report,
    pr_curve, plot_pr_curve, get_roc_auc_analysis, plot_training_history)


# ============================================================================
#  Helpers d'affichage
# ============================================================================

def step(n, titre):
    print("\n" + "=" * 70)
    print(f"  ÉTAPE {n} — {titre}")
    print("=" * 70)

def describe_dataset(df, nom):
    """Stats d'un ensemble : taille, équilibre des classes, ventilation
    classe × côté × projection."""
    n = len(df)
    print(f"\n  [{nom}] — {n} images")
    if n == 0:
        print("     ⚠️  ENSEMBLE VIDE")
        return

    d = df.copy()
    d['main'] = d['laterality'].map({'L': 'gauche', 'R': 'droite'}).fillna('inconnu')
    d['vue']  = d['projection'].map({1: 'frontale', 2: 'laterale'}).fillna('autre')

    for classe in sorted(d['fracture_visible'].fillna(0).unique()):
        libelle = "fracture" if classe == 1 else "sain    "
        c = (d['fracture_visible'].fillna(0) == classe).sum()
        print(f"     {libelle} : {c:5d}  ({c/n:5.1%})")

    print("     " + "-" * 48)
    print(f"     {'classe':10s}{'main':9s}{'vue':11s}{'n':>6s}{'%':>8s}")
    frac = d['fracture_visible'].fillna(0)
    for classe in sorted(frac.unique()):
        libelle = "fracture" if classe == 1 else "sain"
        sub = d[frac == classe]
        grp = sub.groupby(['main', 'vue']).size().sort_index()
        for (main, vue), c in grp.items():
            print(f"     {libelle:10s}{main:9s}{vue:11s}{c:6d}{c/n:8.1%}")

# ============================================================================
#  ZONE DE LUCA — chargement
# ============================================================================
def initialisation():
    t_start = time.time()

    if Path(BASE_DIR).name != "radio_ai":
        sys.exit("ATTENTION : erreur de localisation radio_ai")

    step(1, "Chargement des données")
    t0 = time.time()
    df = load_data()
    print(f"  ⏱  chargement df : {time.time() - t0:.1f}s")

    print(f"\n  Dataset brut chargé : {len(df)} lignes, {df.shape[1]} colonnes")
    print(f"  Mode : {DATA_MODE}  |  colonne chemins : {PATH_COL}")
    print(f"  Exemple de chemin   : ...{df[PATH_COL].sample(1).str[-40:].values[0]}")
    describe_dataset(df, "dataset brut")

    return df, t_start

# ============================================================================
#  ZONE DE MODIBO / MERWAN — preprocessing et modèle
# ============================================================================

def preproc(df):
    step(2, "Filtrage des cas exploitables")
    df_filtered_CNN = filtering(df)

    retenus = len(df_filtered_CNN)
    ecartes = len(df) - retenus
    print(f"\n  Retenus : {retenus}   |   Écartés : {ecartes}  ({ecartes / len(df):.1%} du brut)")
    describe_dataset(df_filtered_CNN, "après filtrage")


    step(3, "Découpage train / val / test")
    # UN SEUL appel à preprocessing (sinon on écrase le cache du warmup)
    (train_ds, val_ds, test_ds), (data_train, data_val, data_test) = preprocessing(df_filtered_CNN)

    print(f"\n  Remplissage du cache (mode : {DATA_MODE}) :")
    warmup_cache(train_ds, "train")
    warmup_cache(val_ds, "val")

    describe_dataset(data_train, "TRAIN")
    describe_dataset(data_val,   "VAL")
    describe_dataset(data_test,  "TEST")

    train_ids = set(data_train['patient_id'])
    val_ids   = set(data_val['patient_id'])
    test_ids  = set(data_test['patient_id'])

    print("\n  Vérification split par patient :")
    print(f"    patients train : {len(train_ids)}")
    print(f"    patients val   : {len(val_ids)}")
    print(f"    patients test  : {len(test_ids)}")
    print(f"    overlap train∩val  : {len(train_ids & val_ids)}")
    print(f"    overlap train∩test : {len(train_ids & test_ids)}")
    print(f"    overlap val∩test   : {len(val_ids & test_ids)}")

    show_train_samples(data_train)

    return (train_ds, val_ds, test_ds), (data_train, data_val, data_test)

def run_model_and_eval(train_ds, val_ds, test_ds, data_train, data_test):

    step(4, "Modèle : entraînement ou chargement")

    if RUN_MODE == "train":
        model = initialize_model()
        model = compile_model(model)
        model.summary()
        model, history = train_model(model, train_ds, val_ds,
                                    y_train=data_train['fracture_visible'])
        save_model(model)   # sauvegarde UNIQUE, seulement après entraînement

    elif RUN_MODE == "eval":
        model = load_model_from_bucket(MODEL_TO_LOAD)   # None -> le plus récent
        print("  Mode évaluation : modèle chargé, pas d'entraînement")

    else:
        sys.exit(f"RUN_MODE inconnu : {RUN_MODE!r} (attendu 'train' ou 'eval')")


    step(5, "Évaluation sur le test")
    results = evaluate_model(model, test_ds)   # UNE SEULE évaluation
    show_predictions(model, data_test, test_ds)

    return model, history

# ============================================================================
#  ZONE DE MARIANA — performance détaillée
# ============================================================================

# looking at how the model is behaving
plot_training_history(history)


# seuil demandé UNE fois, puis propagé à toutes les métriques
threshold = get_user_threshold(default=0.5)
preds = get_predictions(model, test_ds)               # probabilités continues
y_test = get_y_test(test_ds)                           # étiquettes réelles (même ordre)
binary_preds = get_binary_predictions(preds, threshold)
def visualisation_metriques(model, test_ds):


    # seuil demandé UNE fois, puis propagé à toutes les métriques
    threshold = get_user_threshold(default=0.5)
    preds = get_predictions(model, test_ds)               # probabilités continues
    y_test = get_y_test(test_ds)                           # étiquettes réelles (même ordre)
    binary_preds = get_binary_predictions(preds, threshold)

    # métriques sur prédictions binarisées
    get_confusion_matrix(y_test, binary_preds)
    confusion_matrix_display_predicted(y_test, binary_preds)
    comparing_metrics_predictions(y_test, binary_preds)
    get_classification_report(y_test, binary_preds)

    # courbes sur probabilités continues (jamais binarisées)
    pr_curve(y_test, preds)
    plot_pr_curve(y_test, preds)
    get_roc_auc_analysis(y_test, preds)

# ============================================================================
#  ZONE DE LUCA — recap & fin
# ============================================================================

def conclusion(t_start):

    print("\n" + "=" * 70)
    print("  PIPELINE TERMINÉ")
    total = time.time() - t_start
    print(f"  Temps total du run : {total:.1f}s  ({total/60:.1f} min)  |  mode : {DATA_MODE}")
    print("=" * 70)


if __name__ == '__main__':
    df, t_start = initialisation()
    (train_ds, val_ds, test_ds), (data_train, data_val, data_test) = preproc(df)
    model, history = run_model_and_eval(train_ds, val_ds, test_ds, data_train, data_test)
    visualisation_metriques(model, test_ds)
    conclusion(t_start)
