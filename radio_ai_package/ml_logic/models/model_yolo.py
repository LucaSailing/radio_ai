# model_yolo.py — modèle YOLO : init, train, evaluate, save, checkpoints GCS (radio_ai)
#
# Évaluation ramenée au niveau IMAGE pour être comparable au CNN
# (mêmes métriques : accuracy, precision, recall, auc).

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from ultralytics import YOLO   # la CLASSE, séparée de la constante YOLO_WEIGHTS
from google.cloud import storage

from radio_ai_package.params import (
    RAW_DATA_DIR, EPOCH_YOLO, IMAGE_SIZE_YOLO, PATIENCE_YOLO,
    BATCH_SIZE_YOLO, YOLO_WEIGHTS, THRESHOLD,
    DEVICE_YOLO, WORKERS_YOLO, SAVE_PERIOD_YOLO, RUN_NAME_YOLO,
    BUCKET_NAME, MODEL_BUCKET_PREFIX_YOLO, CKPT_BUCKET_PREFIX_YOLO,
)


# fichiers du run à archiver sur GCS (poids + métriques + courbes) ;
# les images de batch sont lourdes et sans valeur d'archive -> ignorées.
_RUN_KEEP_PATTERNS = ("*.pt", "*.csv", "*.png", "*.yaml", "*.json")
_RUN_SKIP_PREFIXES = ("train_batch", "val_batch", "labels")


# =========================================================================
# Helpers
# =========================================================================

def _resolve_device(device):
    """'auto' -> 'cuda' si un GPU est visible, sinon 'cpu'. Toute autre valeur passe telle quelle."""
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"   # "0" fonctionne aussi côté ultralytics


def _bucket():
    """Client GCS + bucket (un seul point de création, réutilisé partout)."""
    return storage.Client().bucket(BUCKET_NAME)


# =========================================================================
# Init / Train
# =========================================================================

def initialize_model_yolo():
    return YOLO(YOLO_WEIGHTS)


def train_model_yolo(
    model: YOLO,
    yaml_path,
    epochs=EPOCH_YOLO,
    imgsz=IMAGE_SIZE_YOLO,
    batch=BATCH_SIZE_YOLO,
    device=DEVICE_YOLO,
    workers=WORKERS_YOLO,
    patience=PATIENCE_YOLO,
    save_period=SAVE_PERIOD_YOLO,
    run_name=RUN_NAME_YOLO,
    resume=False,
):
    device = _resolve_device(device)

    if resume:
        # Reprise : Ultralytics relit data/epochs/device/optimizer depuis le
        # checkpoint. On ne repasse QUE resume=True — surcharger les args ici
        # entrerait en conflit avec l'état sauvegardé.
        results = model.train(resume=True)
    else:
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            workers=workers,
            patience=patience,
            save_period=save_period,   # sauvegarde régulière
            project=str(RAW_DATA_DIR / "yolo_runs"),
            name=run_name,
            exist_ok=True,             # dossier de run fixe (pas de -2, -3…) pour retrouver last.pt
            deterministic=False,
        )

    mode = "repris" if resume else "démarré"
    print(f"✅ Entraînement YOLO {mode} terminé (device={device}, save_period={save_period})")
    return model, results


# =========================================================================
# Évaluation (niveau image, comparable au CNN)
# =========================================================================

def evaluate_model_yolo(
    model, test_img_dir, test_lbl_dir,
    threshold=THRESHOLD,
    imgsz=IMAGE_SIZE_YOLO,   # même résolution que l'entraînement -> comparaison honnête
    device=DEVICE_YOLO,
    conf_floor=0.001,
):
    """
    Évalue YOLO au niveau IMAGE pour le rendre comparable au CNN.
    Reconstruit les MÊMES métriques que evaluate_model (CNN) :
    accuracy, precision, recall, auc.

      y_true  = 1 si le .txt test a >= 1 box fracture, sinon 0
      y_score = confiance MAX des box détectées (0 si aucune)  <-> proba sigmoïde du CNN
      y_pred  = 1 si y_score >= threshold

    Retourne (metrics: dict, df_eval: DataFrame).
    """
    device = _resolve_device(device)
    test_img_dir = Path(test_img_dir)
    test_lbl_dir = Path(test_lbl_dir)

    # conf_floor bas : on veut TOUS les candidats pour une AUC honnête,
    # le seuil de décision s'applique après, sur y_score.
    results = model.predict(
        source=str(test_img_dir),
        conf=conf_floor,
        imgsz=imgsz,
        device=device,
        stream=True,
        verbose=False,
    )

    rows = []
    for r in results:
        stem = Path(r.path).stem
        lbl_path = test_lbl_dir / f"{stem}.txt"
        y_true = 1 if (lbl_path.exists() and lbl_path.read_text().strip()) else 0

        confs = r.boxes.conf
        y_score = float(confs.max()) if confs is not None and len(confs) else 0.0

        rows.append({"filestem": stem, "y_true": y_true, "y_score": y_score})

    df_eval = pd.DataFrame(rows)
    df_eval["y_pred"] = (df_eval["y_score"] >= threshold).astype(int)

    y_true = df_eval["y_true"].to_numpy()
    y_pred = df_eval["y_pred"].to_numpy()
    y_score = df_eval["y_score"].to_numpy()

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else float("nan"),
    }

    print(f"\nÉvaluation YOLO (niveau image, seuil={threshold}, device={device}) :")
    for name, value in metrics.items():
        print(f"  {name:12s} : {value:.4f}")

    return metrics, df_eval


# =========================================================================
# Sauvegarde du modèle final sur GCS (models/yolo — immuable, horodaté)
# =========================================================================

def save_yolo_weights(best_weights_path, name="yolov8n_fracture", timestamp=None,
                      upload_timeout=600):
    """Sauvegarde le SEUL best.pt sur GCS, horodaté (miroir de save_model côté CNN).
    timestamp partageable avec save_yolo_run pour apparier poids et dossier."""
    best_weights_path = Path(best_weights_path)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    blob_name = f"{MODEL_BUCKET_PREFIX_YOLO}/{name}_{timestamp}.pt"

    _bucket().blob(blob_name).upload_from_filename(str(best_weights_path), timeout=upload_timeout)

    gs_uri = f"gs://{BUCKET_NAME}/{blob_name}"
    print(f"  Poids YOLO sauvegardés : {gs_uri}")
    return gs_uri


def save_yolo_run(run_dir, name="yolov8n_fracture", timestamp=None, upload_timeout=600):
    """Sauvegarde les fichiers UTILES du run (poids + métriques + courbes) sur GCS.
    Filtre les images de batch (lourdes, sans valeur d'archive) et allonge le
    timeout d'upload. Un upload qui échoue est ignoré, pas fatal.
    timestamp partageable avec save_yolo_weights."""
    run_dir = Path(run_dir)
    timestamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = f"{MODEL_BUCKET_PREFIX_YOLO}/{name}_{timestamp}"

    bucket = _bucket()
    n, skipped = 0, 0
    for f in run_dir.rglob("*"):
        if not f.is_file():
            continue
        # skip des images de batch lourdes
        if any(f.name.startswith(p) for p in _RUN_SKIP_PREFIXES):
            skipped += 1
            continue
        # ne garder que les extensions utiles
        if not any(f.match(pat) for pat in _RUN_KEEP_PATTERNS):
            skipped += 1
            continue
        rel = f.relative_to(run_dir)
        try:
            bucket.blob(f"{prefix}/{rel}").upload_from_filename(str(f), timeout=upload_timeout)
            n += 1
        except Exception as e:
            print(f"⚠️  Upload de {rel} échoué (ignoré) : {e}")

    gs_uri = f"gs://{BUCKET_NAME}/{prefix}/"
    print(f"  Run YOLO sauvegardé ({n} fichiers, {skipped} ignorés) : {gs_uri}")
    return gs_uri


# =========================================================================
# Checkpoints reprenables sur GCS (checkpoints/yolo — mutable, écrasé)
# =========================================================================

def _ckpt_blob_name(run_name=RUN_NAME_YOLO):
    """Chemin GCS FIXE du checkpoint reprenable (écrasé à chaque upload)."""
    return f"{CKPT_BUCKET_PREFIX_YOLO}/{run_name}/last.pt"


def upload_checkpoint_to_gcs(local_last_pt, run_name=RUN_NAME_YOLO, upload_timeout=600):
    """Pousse le last.pt courant vers un chemin GCS FIXE (écrase le précédent)."""
    local_last_pt = Path(local_last_pt)
    if not local_last_pt.exists():
        return None
    _bucket().blob(_ckpt_blob_name(run_name)).upload_from_filename(
        str(local_last_pt), timeout=upload_timeout
    )
    return f"gs://{BUCKET_NAME}/{_ckpt_blob_name(run_name)}"


def download_checkpoint_from_gcs(dest_path, run_name=RUN_NAME_YOLO):
    """Télécharge le last.pt depuis GCS vers dest_path si présent. Retourne le
    chemin local si téléchargé, sinon None."""
    blob = _bucket().blob(_ckpt_blob_name(run_name))
    if not blob.exists():
        return None
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest_path))
    return dest_path


def delete_checkpoint_from_gcs(run_name=RUN_NAME_YOLO):
    """Supprime le checkpoint reprenable sur GCS (à appeler après un run terminé
    avec succès, pour éviter de proposer une reprise impossible au prochain run)."""
    blob = _bucket().blob(_ckpt_blob_name(run_name))
    if blob.exists():
        blob.delete()
        print(f"🧹 Checkpoint GCS supprimé (run terminé) : gs://{BUCKET_NAME}/{_ckpt_blob_name(run_name)}")


def attach_gcs_checkpoint_callback(model, run_name=RUN_NAME_YOLO, save_period=SAVE_PERIOD_YOLO):
    """Uploade last.pt vers GCS tous les `save_period` epochs (et au dernier).
    Sauvegarde AUSSI, en local, une copie horodatée du last.pt (par epoch) ET le
    best.pt courant, AVANT l'upload — pour ne jamais dépendre de GCS.
    on_model_save est appelé à CHAQUE epoch — on filtre nous-mêmes."""
    import shutil

    # Dossier local de secours (défini ici : aucune constante externe requise)
    local_ckpt_dir = RAW_DATA_DIR / "yolo_runs" / "_checkpoints_backup" / run_name

    def _on_model_save(trainer):
        epoch = getattr(trainer, "epoch", 0) + 1   # epoch est 0-indexé
        is_last = epoch >= getattr(trainer, "epochs", epoch)
        if not (save_period > 0 and (epoch % save_period == 0 or is_last)):
            return

        last_pt = getattr(trainer, "last", None) or Path(trainer.save_dir) / "weights" / "last.pt"
        last_pt = Path(last_pt)

        # --- 1) Sauvegarde LOCALE horodatée du last.pt (ne dépend d'aucun réseau) ---
        try:
            if last_pt.exists():
                local_ckpt_dir.mkdir(parents=True, exist_ok=True)
                dst = local_ckpt_dir / f"epoch{epoch:03d}.pt"
                shutil.copy(last_pt, dst)
                print(f"💾 Checkpoint LOCAL (epoch {epoch}) : {dst}")
        except Exception as e:
            print(f"⚠️  Sauvegarde locale du last échouée (non bloquant) : {e}")

        # --- 1bis) Sauvegarde LOCALE du best.pt (meilleur modèle à cet instant) ---
        try:
            best_pt = last_pt.parent / "best.pt"
            if best_pt.exists():
                local_ckpt_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(best_pt, local_ckpt_dir / "best.pt")
                print(f"💾 Best LOCAL (epoch {epoch}) : {local_ckpt_dir / 'best.pt'}")
        except Exception as e:
            print(f"⚠️  Sauvegarde locale du best échouée (non bloquant) : {e}")

        # --- 2) Upload GCS du last.pt (comportement d'origine, inchangé) ---
        try:
            uri = upload_checkpoint_to_gcs(last_pt, run_name)
            if uri:
                print(f"⬆️  Checkpoint GCS (epoch {epoch}) : {uri}")
        except Exception as e:
            print(f"⚠️  Upload checkpoint GCS échoué (non bloquant) : {e}")

    model.add_callback("on_model_save", _on_model_save)
    return model

def load_yolo_model_from_bucket(
    blob_name: str = None,
    local_dir: str = "/tmp/yolo_models",
) -> YOLO:
    """Downloads a YOLO model weight file (.pt) from GCS and instantiates the Ultralytics YOLO model.

    If `blob_name` is None, it dynamically fetches the most recently updated .pt model
    from the MODEL_BUCKET_PREFIX_YOLO path.

    Parameters:
    -----------
    blob_name : str, optional
        Specific path/filename in GCS bucket (e.g., 'models/yolo/yolov8n_fracture_20260831.pt').
    local_dir : str, optional
        Local cache directory to save the downloaded model weights before loading.

    Returns:
    --------
    YOLO : Loaded Ultralytics YOLO model instance.
    """
    bucket = storage.Client().bucket(BUCKET_NAME)

    # 1. If no specific blob name is provided, automatically locate the latest .pt model
    if not blob_name:
        blobs = list(
            bucket.list_blobs(prefix=f"{MODEL_BUCKET_PREFIX_YOLO}/")
        )
        pt_blobs = [b for b in blobs if b.name.endswith(".pt")]

        if not pt_blobs:
            raise FileNotFoundError(
                f"No '.pt' model checkpoints found in gs://{BUCKET_NAME}/{MODEL_BUCKET_PREFIX_YOLO}/"
            )

        # Sort blobs by updated timestamp to grab the latest trained model
        pt_blobs.sort(key=lambda b: b.updated, reverse=True)
        target_blob = pt_blobs[0]
        print(f"🔍 Found latest YOLO model on GCS: gs://{BUCKET_NAME}/{target_blob.name}")
    else:
        target_blob = bucket.blob(blob_name)
        if not target_blob.exists():
            raise FileNotFoundError(
                f"Model blob 'gs://{BUCKET_NAME}/{blob_name}' does not exist."
            )

    # 2. Setup local destination path
    local_path = Path(local_dir) / Path(target_blob.name).name
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Download weights from GCS if not already cached locally
    if not local_path.exists():
        print(f"⬇️ Downloading YOLO weights from GCS to local path: {local_path}...")
        target_blob.download_to_filename(str(local_path))
        print("✅ Download completed.")
    else:
        print(f"⚡ Using locally cached model weights at: {local_path}")

    # 4. Instantiate and return the YOLO model instance
    return YOLO(str(local_path))
