import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from radio_ai_package.params import IMG_SIZE, RAW_DATA_DIR, PATH_COL

VIZ_DIR = RAW_DATA_DIR / "viz"


# ---------- helpers partagés ----------

def _load_for_display(path, contrast_stretch=True):
    """Recharge une image avec le MÊME preprocessing que l'entraînement
    (padding préservant le ratio + normalisation). Lit indifféremment un
    chemin local ou une URI gs:// (tf.io.read_file gère les deux)."""
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize_with_pad(img, IMG_SIZE[0], IMG_SIZE[1])
    img = tf.cast(img, tf.float32) / 255.0
    arr = img.numpy().squeeze()

    if contrast_stretch:
        lo, hi = np.percentile(arr[arr > 0], [2, 98]) if (arr > 0).any() else (0, 1)
        arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1)
    return arr


def _label(row):
    """Libellé lisible : classe | main | vue."""
    classe = "FRACTURE" if row['fracture_visible'] == 1 else "sain"
    main = {'L': 'gauche', 'R': 'droite'}.get(row['laterality'], '?')
    vue = {1: 'frontale', 2: 'latérale'}.get(row['projection'], '?')
    return f"{classe} | {main} | {vue}"


def _resolve_path_col(df, path_col):
    """Choisit la colonne de chemins : celle demandée si présente, sinon
    bascule automatiquement sur l'autre (local <-> distant)."""
    if path_col in df.columns:
        return path_col
    for fallback in ("file_path", "file_path_gs"):
        if fallback in df.columns:
            return fallback
    raise KeyError("Aucune colonne de chemins trouvée (file_path / file_path_gs).")


# ---------- 1. échantillon de train ----------

def show_train_samples(data_train, n=12, cmap='bone', seed=42, contrast=True,
                       path_col=PATH_COL):
    """12 images de train (2 lignes de 6) telles que le modèle les voit
    (padding visible), avec classe / main / vue en titre."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    col = _resolve_path_col(data_train, path_col)

    sample = data_train.sample(min(n, len(data_train)), random_state=seed)

    cols = 6
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.8))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        img = _load_for_display(row[col], contrast_stretch=contrast)
        ax.imshow(img, cmap=cmap)
        couleur = 'crimson' if row['fracture_visible'] == 1 else 'seagreen'
        ax.set_title(_label(row), fontsize=8, color=couleur)
        ax.axis('off')
    for ax in axes[len(sample):]:
        ax.axis('off')

    fig.suptitle("Échantillon TRAIN (padding préservant le ratio)", fontsize=11)
    plt.tight_layout()
    path = VIZ_DIR / "train_samples.png"
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure enregistrée : {path}")
    return path


# ---------- 2. prédictions test : bien vs mal prédites ----------

def show_predictions(model, data_test, test_ds, n_each=6, cmap='bone',
                     seed=42, contrast=True, threshold=0.5, path_col=PATH_COL):
    """6 images bien prédites (ligne 1) et 6 mal prédites (ligne 2) du test,
    avec vérité / prédiction / proba. Vert = correct, rouge = erreur."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    col = _resolve_path_col(data_test, path_col)

    proba = model.predict(test_ds, verbose=0).squeeze()

    d = data_test.copy().reset_index(drop=True)
    if len(proba) != len(d):
        print(f"  ⚠️  désalignement : {len(proba)} prédictions vs {len(d)} lignes")
        return None

    d['proba'] = proba
    d['pred'] = (d['proba'] >= threshold).astype(int)
    d['truth'] = d['fracture_visible'].astype(int)
    d['correct'] = d['pred'] == d['truth']

    bien = d[d['correct']].sample(min(n_each, d['correct'].sum()), random_state=seed)
    mal  = d[~d['correct']].sample(min(n_each, (~d['correct']).sum()), random_state=seed)
    sample = pd.concat([bien, mal])

    cols = n_each
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2.2, 2 * 3.0))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        img = _load_for_display(row[col], contrast_stretch=contrast)
        ax.imshow(img, cmap=cmap)
        vrai = "FRACTURE" if row['truth'] == 1 else "sain"
        prevu = "FRACTURE" if row['pred'] == 1 else "sain"
        couleur = 'seagreen' if row['correct'] else 'crimson'
        ax.set_title(f"vrai: {vrai}\nprévu: {prevu} ({row['proba']:.2f})",
                     fontsize=8, color=couleur)
        ax.axis('off')
    for ax in axes[len(sample):]:
        ax.axis('off')

    fig.suptitle("TEST — ligne 1 : bien prédites | ligne 2 : mal prédites", fontsize=11)
    plt.tight_layout()
    path = VIZ_DIR / "test_predictions.png"
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure enregistrée : {path}")
    return path
