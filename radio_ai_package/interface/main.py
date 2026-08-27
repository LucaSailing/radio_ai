# ============================================================================
#  main.py — pipeline détection de fractures (radio_ai)
# ============================================================================

import os

# doit être défini AVANT l'import de tensorflow / google
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"        # masque les logs C++ de TF (oneDNN, CPU, cuInit)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"        # coupe le message oneDNN
os.environ["GRPC_VERBOSITY"] = "ERROR"           # calme les logs gRPC/absl (I0000...)
os.environ["GLOG_minloglevel"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # pas de GPU -> TF ne cherche pas les drivers CUDA

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)   # masque le FutureWarning


# Imports du main
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from radio_ai_package.params import *

# Imports de Luca (data)
from radio_ai_package.ml_logic.data import load_df_with_local_paths, import_data_bucket
from radio_ai_package.ml_logic.preprocessors.preprocessor_CNN import filtering, preprocessing
from radio_ai_package.ml_logic.viz import show_train_samples, show_predictions

# Imports de Modibo / Merwan (modèle)
from radio_ai_package.ml_logic.models.model_CNN import (initialize_model, compile_model,
                                                        train_model, evaluate_model, save_model)

# Imports de Mariana (performance metrics)
from radio_ai_package.ml_logic.models.model_CNN import get_user_threshold, get_predictions, get_x_test, get_y_test, get_binary_predictions
from radio_ai_package.ml_logic.models.model_CNN import get_confusion_matrix, confusion_matrix_display_predicted
from radio_ai_package.ml_logic.models.model_CNN import comparing_metrics_predictions, get_classification_report
from radio_ai_package.ml_logic.models.model_CNN import pr_curve, plot_pr_curve, get_roc_auc_analysis

# ============================================================================
#  Helpers d'affichage — factorisent le reporting pour un flow lisible
# ============================================================================

def step(n, titre):
    """Affiche un séparateur d'étape bien visible."""
    print("\n" + "=" * 70)
    print(f"  ÉTAPE {n} — {titre}")
    print("=" * 70)


def describe_dataset(df, nom):
    """Stats d'un ensemble : taille, équilibre des classes, et ventilation
    classe × côté (laterality) × projection (frontale/latérale)."""
    n = len(df)
    print(f"\n  [{nom}] — {n} images")
    if n == 0:
        print("     ⚠️  ENSEMBLE VIDE")
        return

    d = df.copy()
    d['main'] = d['laterality'].map({'L': 'gauche', 'R': 'droite'}).fillna('inconnu')
    d['vue']  = d['projection'].map({1: 'frontale', 2: 'laterale'}).fillna('autre')

    # équilibre global des classes
    for classe in sorted(d['fracture_visible'].fillna(0).unique()):
        libelle = "fracture" if classe == 1 else "sain    "
        c = (d['fracture_visible'].fillna(0) == classe).sum()
        print(f"     {libelle} : {c:5d}  ({c/n:5.1%})")

    # croisement classe × main × projection
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
#  ZONE DE LUCA — chargement des données
# ============================================================================

# garde-fou : on doit être à la racine du package
if Path(BASE_DIR).name != "radio_ai":
    sys.exit("ATTENTION : erreur de localisation radio_ai")

step(1, "Chargement des données") # fonction d'affichage dans le terminal de l'avancement

# df = load_df_with_local_paths() # voie rapide : images déjà en local
df = import_data_bucket()       # voie complète : télécharge depuis bucket (skippe si fichier existe déja en local)

print(f"\n  Dataset brut chargé : {len(df)} lignes, {df.shape[1]} colonnes")
print(f"  Exemple de chemin   : ...{df.file_path.sample(1).str[-40:].values[0]}")
describe_dataset(df, "dataset brut")


# ============================================================================
#  ZONE DE MODIBO / MERWAN — preprocessing et modèle
# ============================================================================

step(2, "Filtrage des cas exploitables") # debut du modele de CNN

df_filtered_CNN = filtering(df)

# combien on garde / combien on écarte, et pourquoi c'est visible
retenus = len(df_filtered_CNN)
ecartes = len(df) - retenus
print(f"\n  Retenus : {retenus}   |   Écartés : {ecartes}  ({ecartes / len(df):.1%} du brut)")
describe_dataset(df_filtered_CNN, "après filtrage")


step(3, "Découpage train / val / test")

# preprocessing renvoie les 3 datasets tf.data + les 3 DataFrames
(train_ds, val_ds, test_ds), (data_train, data_val, data_test) = preprocessing(df_filtered_CNN)

# pour pouvoir en afficher les stats. Adapte selon ce que preprocessing retourne.
describe_dataset(data_train, "TRAIN")
describe_dataset(data_val,   "VAL")
describe_dataset(data_test,  "TEST")

train_ids = set(data_train['patient_id'])
val_ids   = set(data_val['patient_id'])
test_ids  = set(data_test['patient_id'])

# pour vérifier le split par patient et qu'il n'y ait pas du leackage.
print("\n  Vérification split par patient :")
print(f"    patients train : {len(train_ids)}")
print(f"    patients val   : {len(val_ids)}")
print(f"    patients test  : {len(test_ids)}")
print(f"    overlap train∩val  : {len(train_ids & val_ids)}")
print(f"    overlap train∩test : {len(train_ids & test_ids)}")
print(f"    overlap val∩test   : {len(val_ids & test_ids)}")


# montre des images de train - 1er draft - fonction à améliorer
show_train_samples(data_train)


step(4, "Construction et entraînement du modèle")

model = initialize_model()
model = compile_model(model)
model.summary()

model, history = train_model(model, train_ds, val_ds,
                             y_train=data_train['fracture_visible'])









# ============================================================================
#  ZONE DE MARIANA — performance du modèle
# ============================================================================

step(5, "Évaluation sur le test")

results = evaluate_model(model, test_ds)
save_model(model)

# Generating predictions
preds = get_predictions(test_ds)

# Getting the threshold
threshold = get_user_threshold(test_ds)

# Defyning x_test
X_test = get_x_test(test_ds)

# Defyning y_test
y_test = get_y_test(test_ds)

# Generating binary predictions (fracture vs no fracture)
binary_predictions = get_binary_predictions(preds)

# Generating the confusion matrix
confusion_matrix = get_confusion_matrix(y_test, preds)
confusion_matrix

cm_drawing = confusion_matrix_display_predicted(y_test, preds)
cm_drawing

# Comparing metrix
accuracy, precision, recall, f1 = comparing_metrics_predictions(y_test, preds)
accuracy, precision, recall, f1

# Classification report
classification_report = get_classification_report(y_test, preds)
classification_report

# Precision Recall Curve
precision_recall_curve = pr_curve(y_test, preds)
precision_recall_curve

plot_pr_curve(y_test, preds)

# ROC_AUC
fpr, tpr, thresholds, auc_score, best_threshold_j=get_roc_auc_analysis(y_test, preds)
fpr, tpr, thresholds, auc_score, best_threshold_j

print("\n" + "=" * 70)
print("  PIPELINE TERMINÉ")
print("=" * 70)

# montre des images de predictions - 1er draft - fonction à améliorer
show_predictions(model, data_test, test_ds)
