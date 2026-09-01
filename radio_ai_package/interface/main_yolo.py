"""
main_yolo.py — pipeline de détection de fractures par YOLO (radio_ai)
Miroir de main.py (CNN) : chargement des données, préparation du dataset YOLO,
entraînement (reprenable, checkpoints synchronisés sur GCS), évaluation et
rapport de suivi.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # coupe oneDNN / cpu_feature_guard / cudart_stub
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # coupe le message oneDNN + non-déterminisme
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")

from pathlib import Path

from ultralytics import YOLO

from radio_ai_package.params import RAW_DATA_DIR, RUN_NAME_YOLO, IMAGE_SIZE_YOLO
from radio_ai_package.ml_logic.data import load_data  # adapte le nom si besoin
from radio_ai_package.ml_logic.preprocessors.preprocessor_yolo import build_yolo_dataset
from radio_ai_package.ml_logic.models.model_yolo import (
    initialize_model_yolo,
    train_model_yolo,
    evaluate_model_yolo,
    save_yolo_weights,
    save_yolo_run,
    download_checkpoint_from_gcs,
    delete_checkpoint_from_gcs,
    attach_gcs_checkpoint_callback,
)
from radio_ai_package.ml_logic.reporting.report_yolo import save_report_to_gcs


# ============================================================================
#  Reprise (checkpoint local ou GCS)
# ============================================================================

def _local_last_pt(run_name=RUN_NAME_YOLO):
    """Chemin local du last.pt du run."""
    return RAW_DATA_DIR / "yolo_runs" / run_name / "weights" / "last.pt"


def _is_resumable(ckpt_path):
    """Un last.pt n'est reprenable que s'il garde l'état d'optimizer/epoch
    (run INTERROMPU). Après un run TERMINÉ, Ultralytics strippe l'optimizer :
    'resume=True' échouerait et repartirait en douce sur coco8/100 epochs.
    On détecte ce cas ici pour ne jamais proposer une reprise impossible."""
    try:
        import torch
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ck.get("optimizer") is not None and ck.get("epoch", -1) >= 0
    except Exception:
        return False


def _prepare_resume(run_name=RUN_NAME_YOLO):
    """
    Détermine s'il faut reprendre, en cherchant un checkpoint local PUIS sur GCS,
    puis en VÉRIFIANT qu'il est réellement reprenable. Retourne le chemin local
    du checkpoint à reprendre, ou None si run frais.

    Non interactif (pas de stdin) : run FRAIS par défaut.
    """
    local = _local_last_pt(run_name)

    if not local.exists():
        # Local vide (VM neuve ?) -> tenter GCS
        restored = download_checkpoint_from_gcs(local, run_name)
        if restored is not None:
            print(f"⬇️  Checkpoint restauré depuis GCS : {restored}")
        else:
            return None  # rien nulle part -> run frais

    # Un checkpoint local existe (d'origine ou restauré) : est-il reprenable ?
    if not _is_resumable(local):
        print(f"ℹ️  Checkpoint présent mais non reprenable (run déjà terminé) → run frais.")
        return None

    # Reprenable -> demander confirmation
    try:
        answer = input(f"⏸  Checkpoint reprenable trouvé ({local}). Reprendre ? [o/N] ").strip().lower()
    except EOFError:
        answer = "n"  # non interactif (GCP startup, CI) -> run FRAIS par défaut
    return local if answer in ("o", "oui", "y", "yes") else None


# ============================================================================
#  Nettoyage post-run (évite toute reprise fantôme au prochain lancement)
# ============================================================================

def _cleanup_checkpoints(run_name=RUN_NAME_YOLO):
    """Supprime le checkpoint reprenable (local + GCS) après un run mené à terme."""
    local = _local_last_pt(run_name)
    if local.exists():
        local.unlink()
        print(f"🧹 Checkpoint local supprimé (run terminé) : {local}")
    try:
        delete_checkpoint_from_gcs(run_name)
    except Exception as e:
        print(f"⚠️  Suppression checkpoint GCS échouée (non bloquant) : {e}")


# ============================================================================
#  Pipeline
# ============================================================================

def train_yolo():
    """Charge les données, prépare le dataset, entraîne (reprise + checkpoints
    GCS), évalue, sauvegarde le modèle et génère le rapport de suivi."""
    df = load_data()

    (yaml_path, dataset_dir), (data_train, data_val, data_test) = build_yolo_dataset(df)

    # --- Reprise éventuelle (local ou GCS, seulement si réellement reprenable) ---
    resume_ckpt = _prepare_resume()

    if resume_ckpt is not None:
        print(f"▶️  Reprise depuis {resume_ckpt}")
        model = YOLO(resume_ckpt)
        model = attach_gcs_checkpoint_callback(model)   # continue de synchroniser GCS
        model, results = train_model_yolo(model, yaml_path, resume=True)
    else:
        model = initialize_model_yolo()
        model = attach_gcs_checkpoint_callback(model)   # synchronise last.pt -> GCS pendant le run
        model, results = train_model_yolo(model, yaml_path, resume=False)

    # --- Évaluation sur les MEILLEURS poids (cohérent avec le CNN) ---
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    model = YOLO(best_weights)
    print(f"📦 Évaluation sur les meilleurs poids : {best_weights}")

    test_img_dir = dataset_dir / "images" / "test"
    test_lbl_dir = dataset_dir / "labels" / "test"
    metrics, df_eval = evaluate_model_yolo(model, test_img_dir, test_lbl_dir)

    # --- Garde-fous de cohérence ---
    if len(df_eval) != len(data_test):
        print(f"⚠️  {len(df_eval)} images évaluées pour {len(data_test)} attendues (split test)")
    if df_eval["y_true"].nunique() < 2:
        print(f"⚠️  Une seule classe dans y_true {df_eval['y_true'].value_counts().to_dict()} → AUC = nan")

    # --- Sauvegarde du modèle final (best.pt) : critique, on la laisse remonter ---
    ts = None
    try:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_yolo_weights(best_weights, timestamp=ts)
    except Exception as e:
        print(f"⚠️  Sauvegarde du best.pt échouée : {e}")

    # --- Sauvegardes accessoires (dossier de run + rapport) : NON fatales ---
    #     un timeout GCS ici ne doit pas empêcher le nettoyage final.
    for step_name, fn in [
        ("run",    lambda: save_yolo_run(results.save_dir, timestamp=ts)),
        ("report", lambda: save_report_to_gcs(
                        model, metrics, df_eval,
                        results_dir=results.save_dir,
                        test_img_dir=test_img_dir,
                        test_lbl_dir=test_lbl_dir,
                        imgsz=IMAGE_SIZE_YOLO)),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"⚠️  Étape '{step_name}' échouée (non bloquant) : {e}")

    # --- Nettoyage : tourne TOUJOURS, même si une sauvegarde accessoire a échoué ---
    _cleanup_checkpoints()

    return model, metrics


if __name__ == "__main__":
    train_yolo()
