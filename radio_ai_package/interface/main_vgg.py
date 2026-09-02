# ============================================================================
#  main_vgg.py — pipeline détection de fractures via VGG16 (radio_ai)
# ============================================================================

import os

# Configuration TensorFlow / Logs (AVANT l'import de TF)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from radio_ai_package.params import *

# --- data (Luca) ---
from radio_ai_package.ml_logic.data import load_data
from radio_ai_package.ml_logic.preprocessors.preprocessor_vgg import build_vgg_dataset
from radio_ai_package.ml_logic.viz import show_predictions, show_train_samples

# --- modèle VGG16 (Modibo / Merwan) ---
from radio_ai_package.ml_logic.models.model_vgg import (
    compile_model_vgg,
    evaluate_model_vgg,
    initialize_model_vgg,
    train_model_vgg,
)
from radio_ai_package.ml_logic.models.model_CNN import (
    save_model,
    load_model_from_bucket,
    warmup_cache,
)

# --- métriques (Mariana) ---
from radio_ai_package.ml_logic.performance_metrics import (
    comparing_metrics_predictions,
    confusion_matrix_display_predicted,
    get_binary_predictions,
    get_classification_report,
    get_confusion_matrix,
    get_predictions,
    get_roc_auc_analysis,
    get_x_test,
    get_y_test,
    plot_pr_curve,
    plot_training_history,
    pr_curve,
)

# --- Grad-CAM ---
from radio_ai_package.ml_logic.grad_cam import plot_gradcam_confusion_matrix


# ============================================================================
#  Helpers d'affichage
# ============================================================================

def step(n, titre):
    print("\n" + "=" * 70)
    print(f"  ÉTAPE {n} — {titre}")
    print("=" * 70)


def describe_dataset(df, nom):
    """Stats d'un ensemble : taille, équilibre des classes, ventilation."""
    n = len(df)
    print(f"\n  [{nom}] — {n} images")
    if n == 0:
        print("     ⚠️  ENSEMBLE VIDE")
        return

    d = df.copy()
    d["main"] = d["laterality"].map({"L": "gauche", "R": "droite"}).fillna("inconnu")
    d["vue"] = d["projection"].map({1: "frontale", 2: "laterale"}).fillna("autre")

    for classe in sorted(d["fracture_visible"].fillna(0).unique()):
        libelle = "fracture" if classe == 1 else "sain    "
        c = (d["fracture_visible"].fillna(0) == classe).sum()
        print(f"     {libelle} : {c:5d}  ({c/n:5.1%})")

    print("     " + "-" * 48)
    print(f"     {'classe':10s}{'main':9s}{'vue':11s}{'n':>6s}{'%':>8s}")
    frac = d["fracture_visible"].fillna(0)
    for classe in sorted(frac.unique()):
        libelle = "fracture" if classe == 1 else "sain"
        sub = d[frac == classe]
        grp = sub.groupby(["main", "vue"]).size().sort_index()
        for (main, vue), c in grp.items():
            print(f"     {libelle:10s}{main:9s}{vue:11s}{c:6d}{c/n:8.1%}")


# ============================================================================
#  ZONE DE LUCA — Chargement & Preprocessing VGG
# ============================================================================

def initialisation():
    t_start = time.time()

    if Path(BASE_DIR).name != "radio_ai":
        sys.exit("ATTENTION : erreur de localisation radio_ai")

    step(1, "Chargement des données")
    t0 = time.time()
    df = load_data()
    print(f"  ⏱  chargement df : {time.time() - t0:.1f}s")
    describe_dataset(df, "dataset brut")

    return df, t_start


def preproc_vgg(df):
    step(2, "Préparation du dataset VGG16 (3 canaux, 224x224)")
    (train_ds, val_ds, test_ds), (data_train, data_val, data_test) = build_vgg_dataset(df)

    print(f"\n  Remplissage du cache (mode : {DATA_MODE}) :")
    warmup_cache(train_ds, "train")
    warmup_cache(val_ds, "val")

    describe_dataset(data_train, "TRAIN (VGG)")
    describe_dataset(data_val, "VAL (VGG)")
    describe_dataset(data_test, "TEST (VGG)")

    show_train_samples(data_train)

    return (train_ds, val_ds, test_ds), (data_train, data_val, data_test)


# ============================================================================
#  ZONE MODIBO / MERWAN — Modèle VGG16
# ============================================================================

def run_model_vgg(train_ds, val_ds, test_ds, data_train, data_test):
    step(3, "Modèle VGG16 : entraînement ou chargement")

    history = None
    if RUN_MODE == "train":
        model = initialize_model_vgg()
        model = compile_model_vgg(model)
        model.summary()

        model, history = train_model_vgg(
            model,
            train_ds,
            val_ds,
            y_train=data_train["fracture_visible"],
        )
        save_model(model)
        plot_training_history(history)

    elif RUN_MODE == "eval":
        model = load_model_from_bucket(MODEL_TO_LOAD)
        print("  Mode évaluation : modèle chargé, pas d'entraînement")

    else:
        sys.exit(f"RUN_MODE inconnu : {RUN_MODE!r}")

    step(4, "Évaluation sur le dataset test")
    results = evaluate_model_vgg(model, test_ds)
    show_predictions(model, data_test, test_ds)

    return model, history


# ============================================================================
#  ZONE MARIANA — Performance & Grad-CAM VGG
# ============================================================================

def visualisation_metriques(model, test_ds):
    step(5, "Analyse détaillée des performances & Grad-CAM")

    threshold = THRESHOLD
    preds = get_predictions(model, test_ds)
    y_test = get_y_test(test_ds)
    binary_preds = get_binary_predictions(preds, threshold)

    # Métriques
    get_confusion_matrix(y_test, binary_preds)
    confusion_matrix_display_predicted(y_test, binary_preds)
    comparing_metrics_predictions(y_test, binary_preds)
    get_classification_report(y_test, binary_preds)

    # Courbes ROC / PR
    pr_curve(y_test, preds)
    plot_pr_curve(y_test, preds)
    get_roc_auc_analysis(y_test, preds)

    # Grille Grad-CAM
    print("\n  Génération de la grille Grad-CAM VGG16 (TP, TN, FP, FN)...")
    try:
        X_test = get_x_test(test_ds)
        plot_gradcam_confusion_matrix(
            model=model,
            X_test=X_test,
            y_test=y_test,
            preds=preds,
            binary_preds=binary_preds,
            alpha=0.4,
            filename="vgg_gradcam_confusion_matrix.png",
        )
    except Exception as e:
        print(f"  ⚠️  Échec génération Grad-CAM VGG : {e}")


# ============================================================================
#  Main Execution
# ============================================================================

def main():
    df, t_start = initialisation()
    (train_ds, val_ds, test_ds), (data_train, data_val, data_test) = preproc_vgg(df)
    model, history = run_model_vgg(train_ds, val_ds, test_ds, data_train, data_test)
    visualisation_metriques(model, test_ds)

    print("\n" + "=" * 70)
    print("  PIPELINE VGG16 TERMINÉ")
    total = time.time() - t_start
    print(f"  Temps total : {total:.1f}s ({total/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
