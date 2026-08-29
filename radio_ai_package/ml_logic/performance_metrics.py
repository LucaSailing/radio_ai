# ============================================================================
#  performance_metrics.py — métriques et courbes de performance (radio_ai)
# ============================================================================

import sys

import matplotlib
matplotlib.use("Agg")            # backend headless (WSL sans écran) -> savefig, pas show
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay, confusion_matrix
                             precision_recall_curve, PrecisionRecallDisplay,
                             roc_curve, roc_auc_score, classification_report)

from radio_ai_package.params import RAW_DATA_DIR

from pathlib import Path
import cv2
import tensorflow as tf

VIZ_DIR = RAW_DATA_DIR / "viz"


# ---------- seuil de décision ----------

def get_user_threshold(default=0.5):
    """Demande un seuil de décision (0.0–1.0). Repli sur `default` si l'entrée
    est vide/invalide, OU si on n'est pas dans un terminal interactif (VM,
    pipeline non interactif) — évite de bloquer le run sur un input()."""
    if not sys.stdin.isatty():
        print(f"  (non interactif) seuil par défaut : {default}")
        return default

    while True:
        user_input = input(f"Seuil de probabilité (0.0 à 1.0) [défaut {default}] : ").strip()
        if not user_input:
            print(f"  -> seuil par défaut : {default}")
            return default
        try:
            threshold = float(user_input)
            if 0.0 <= threshold <= 1.0:
                print(f"  -> seuil retenu : {threshold}")
                return threshold
            print("  seuil hors bornes (0.0–1.0), réessaie.")
        except ValueError:
            print("  format invalide, entre un flottant (ex. 0.35).")


# ---------- prédictions ----------

def get_predictions(model, test_ds):
    """Probabilités continues prédites sur le test. On passe le modèle déjà en
    mémoire (entraîné ou chargé du bucket) au lieu d'en recharger un — cohérent
    avec le pipeline, pas de double chargement."""
    preds = model.predict(test_ds)
    return preds


def get_y_test(test_ds):
    """Étiquettes réelles du test, empilées en 1D. L'ordre suit l'itération du
    dataset ; test_ds n'étant PAS shuffle, il reste aligné avec get_predictions."""
    y_test = np.concatenate([y for _, y in test_ds], axis=0)
    print("  y_test shape :", y_test.shape)
    return y_test


def get_x_test(test_ds):
    """Images du test empilées. Optionnel (inspecter des cas) — non requis par
    les métriques elles-mêmes."""
    X_test = np.concatenate([x.numpy() for x, _ in test_ds], axis=0)
    print("  X_test shape :", X_test.shape)
    return X_test


def get_binary_predictions(preds, threshold=0.5):
    """Binarise les probabilités selon un seuil DONNÉ. Le seuil est décidé une
    seule fois en amont et propagé, pour que toutes les métriques soient
    cohérentes entre elles (pas de seuil redemandé à chaque fonction)."""
    binary_preds = (preds.flatten() > threshold).astype(int)
    return binary_preds


# ---------- matrice de confusion ----------

def get_confusion_matrix(y_test, binary_preds):
    """Matrice de confusion en tableau croisé pandas (prédictions binarisées)."""
    results_df = pd.DataFrame({"actual": np.asarray(y_test).flatten(),
                               "predicted": np.asarray(binary_preds).flatten()})
    return pd.crosstab(results_df['actual'], results_df['predicted'])

def get_confusion_matrix_metrics(y_test, binary_preds):
    """Extracts raw counts (TN, FP, FN, TP) from binary test predictions."""
    ## Because sklearn.metrics.confusion_matrix returns a 2D array
    ## ([[TN, FP], [FN, TP]]), Python sees 2 rows instead of 4 individual numbers
    ##  Flatten the 2D array using .ravel() (or .flatten()) before unpacking:
    y_true = np.asarray(y_test).flatten()
    y_pred = np.asarray(binary_preds).flatten()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp

def get_confusion_matrix_indices(y_test, binary_preds):
    """Returns positional indices for each quadrant of the confusion matrix."""
    y_true = np.asarray(y_test).flatten()
    y_pred = np.asarray(binary_preds).flatten()

    tp_indices = np.where((y_pred == 1) & (y_true == 1))[0]
    tn_indices = np.where((y_pred == 0) & (y_true == 0))[0]
    fp_indices = np.where((y_pred == 1) & (y_true == 0))[0]
    fn_indices = np.where((y_pred == 0) & (y_true == 1))[0]

    return tp_indices, tn_indices, fp_indices, fn_indices

def confusion_matrix_display_predicted(y_test, binary_preds, filename="confusion_matrix.png"):
    """Enregistre la matrice de confusion en PNG (pas de show en WSL)."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay.from_predictions(
        np.asarray(y_test).flatten(),
        np.asarray(binary_preds).flatten(),
        display_labels=['Sain (0)', 'Fracture (1)'],
        cmap=plt.cm.Blues)
    disp.ax_.set_title("Matrice de confusion — test")
    out = VIZ_DIR / filename
    disp.figure_.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(disp.figure_)
    print(f"  Figure enregistrée : {out}")
    return out


# ---------- métriques scalaires ----------

def comparing_metrics_predictions(y_test, binary_preds):
    """Accuracy, precision, recall, F1 (prédictions binarisées)."""
    y_true = np.asarray(y_test).flatten()
    y_pred = np.asarray(binary_preds).flatten()

    accuracy  = round(accuracy_score(y_true, y_pred), 2)
    precision = round(precision_score(y_true, y_pred, zero_division=0), 2)
    recall    = round(recall_score(y_true, y_pred, zero_division=0), 2)
    f1        = round(f1_score(y_true, y_pred, zero_division=0), 2)

    print(f"  Accuracy  = {accuracy}")
    print(f"  Precision = {precision}")
    print(f"  Recall    = {recall}")
    print(f"  F1 score  = {f1}")
    return accuracy, precision, recall, f1


def get_classification_report(y_test, binary_preds):
    """Rapport de classification sklearn (précision/rappel/F1 par classe)."""
    report = classification_report(
        np.asarray(y_test).flatten(),
        np.asarray(binary_preds).flatten(),
        target_names=['Sain (0)', 'Fracture (1)'],
        digits=2, zero_division=0)
    print(report)
    return report


# ---------- courbes (probabilités continues, PAS binarisées) ----------

def pr_curve(y_test, preds):
    """(precision, recall, thresholds) sur probabilités continues."""
    return precision_recall_curve(np.asarray(y_test).flatten(),
                                  np.asarray(preds).flatten())


def plot_pr_curve(y_test, preds, filename="pr_curve.png"):
    """Courbe précision-rappel en PNG. Utilise les probabilités continues
    (surtout pas les prédictions binarisées)."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(
        np.asarray(y_test).flatten(),
        np.asarray(preds).flatten(),
        name="CNN", ax=ax)
    ax.set_title("Courbe précision-rappel")
    ax.grid(True, linestyle="--", alpha=0.6)
    out = VIZ_DIR / filename
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Figure enregistrée : {out}")
    return out


def get_roc_auc_analysis(y_test, preds, filename="roc_curve.png"):
    """ROC-AUC + seuil optimal (Youden's J = TPR - FPR), courbe ROC en PNG.
    Retourne (fpr, tpr, thresholds, auc_score, best_threshold_j)."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_test).flatten()
    y_probs = np.asarray(preds).flatten()

    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold_j = thresholds[best_idx]

    print(f"  AUC : {auc_score:.4f}")
    print(f"  Seuil optimal (Youden's J) : {best_threshold_j:.4f}")
    print(f"  À ce seuil -> TPR {tpr[best_idx]:.4f} | FPR {fpr[best_idx]:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire (0.50)')
    ax.scatter(fpr[best_idx], tpr[best_idx], color='red', s=100, zorder=5,
               label=f'Seuil optimal = {best_threshold_j:.3f}')
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Taux de faux positifs (1 - spécificité)')
    ax.set_ylabel('Taux de vrais positifs (rappel / sensibilité)')
    ax.set_title('Courbe ROC')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    out = VIZ_DIR / filename
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Figure enregistrée : {out}")

    return fpr, tpr, thresholds, auc_score, best_threshold_j


################################### History plot ###############################
def plot_training_history(history, filename="training_history.png"):
    """
    Plots Training vs Validation Loss, Recall, and Accuracy in a 1x3 grid.
    Marks the best epoch based on minimum validation loss.
    """
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.history['loss']) + 1)
    best_epoch = np.argmin(history.history['val_loss']) + 1
    best_val_loss = history.history['val_loss'][best_epoch - 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ==========================================
    # 1. LOSS PLOT
    # ==========================================
    axes[0].plot(epochs, history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
    axes[0].plot(epochs, history.history['val_loss'], label='Val Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    axes[0].axvline(best_epoch, color='red', linestyle=':', label=f'Best Epoch ({best_epoch})')
    axes[0].scatter(best_epoch, best_val_loss, color='red', s=50, zorder=5)

    axes[0].set_title('Binary Crossentropy Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend(frameon=True, loc='upper right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # ==========================================
    # 2. RECALL PLOT
    # ==========================================
    axes[1].plot(epochs, history.history['recall'], label='Train Recall', color='#2ca02c', linewidth=2)
    axes[1].plot(epochs, history.history['val_recall'], label='Val Recall', color='#d62728', linewidth=2, linestyle='--')
    axes[1].axvline(best_epoch, color='red', linestyle=':', label=f'Best Epoch ({best_epoch})')

    axes[1].set_title('Model Recall', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Recall')
    axes[1].legend(frameon=True, loc='lower right')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # ==========================================
    # 3. ACCURACY PLOT
    # ==========================================
    acc_key = 'accuracy' if 'accuracy' in history.history else 'acc'
    val_acc_key = f'val_{acc_key}'

    axes[2].plot(epochs, history.history[acc_key], label='Train Accuracy', color='#9467bd', linewidth=2)
    axes[2].plot(epochs, history.history[val_acc_key], label='Val Accuracy', color='#8c564b', linewidth=2, linestyle='--')
    axes[2].axvline(best_epoch, color='red', linestyle=':', label=f'Best Epoch ({best_epoch})')

    axes[2].set_title('Model Accuracy', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend(frameon=True, loc='lower right')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle('CNN Training & Validation Metrics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out = VIZ_DIR / filename
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.show()
