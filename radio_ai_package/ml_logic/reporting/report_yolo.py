# report_yolo.py — génération du rapport de suivi YOLO sur GCS (radio_ai)

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from radio_ai_package.params import BUCKET_NAME, RUN_NAME_YOLO
from radio_ai_package.ml_logic.models.model_yolo import _bucket

PERF_BUCKET_PREFIX_YOLO = "suivi_perf/yolo"

# couleurs BGR (OpenCV)
COLOR_PRED = (0, 165, 255)   # orange = boîte prédite
COLOR_GT   = (0, 255, 0)     # vert   = vérité terrain (test set)


def _draw_boxes(img, boxes_xywhn, color, label):
    """Dessine des boîtes YOLO normalisées (cx,cy,w,h) sur une image BGR."""
    h, w = img.shape[:2]
    for cx, cy, bw, bh in boxes_xywhn:
        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if boxes_xywhn:
        cv2.putText(img, label, (5, 20 if color == COLOR_PRED else 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def _gt_boxes(lbl_path):
    """Lit les boîtes vérité terrain d'un .txt YOLO (classe cx cy w h)."""
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                boxes.append(tuple(map(float, p[1:])))
    return boxes


def _classify(df_eval):
    """Range chaque image dans un quadrant TP / FN / TN / FP (niveau image)."""
    tp = df_eval[(df_eval.y_true == 1) & (df_eval.y_pred == 1)]
    fn = df_eval[(df_eval.y_true == 1) & (df_eval.y_pred == 0)]
    tn = df_eval[(df_eval.y_true == 0) & (df_eval.y_pred == 0)]
    fp = df_eval[(df_eval.y_true == 0) & (df_eval.y_pred == 1)]
    return {"TP (fracture détectée)": tp, "FN (fracture ratée)": fn,
            "TN (sain OK)": tn, "FP (fausse alerte)": fp}


def build_examples_figure(model, df_eval, test_img_dir, test_lbl_dir,
                          imgsz, n_per_row=5):
    """Planche 4 lignes (TP/FN/TN/FP) × 5 images, boîtes pred (orange) + GT (vert)."""
    test_img_dir = Path(test_img_dir); test_lbl_dir = Path(test_lbl_dir)
    quadrants = _classify(df_eval)

    fig, axes = plt.subplots(4, n_per_row, figsize=(4 * n_per_row, 16))
    fig.suptitle(f"Exemples de prédictions — {RUN_NAME_YOLO}", fontsize=16, y=0.995)

    for row, (title, subdf) in enumerate(quadrants.items()):
        sample = subdf.head(n_per_row)
        for col in range(n_per_row):
            ax = axes[row, col]; ax.axis("off")
            if col == 0:
                ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
            if col >= len(sample):
                continue
            stem = sample.iloc[col]["filestem"]
            img_path = test_img_dir / f"{stem}.png"
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # boîtes prédites (re-inférence sur cette image)
            res = model.predict(source=str(img_path), imgsz=imgsz,
                                conf=0.25, verbose=False)[0]
            pred_boxes = [tuple(b) for b in res.boxes.xywhn.cpu().numpy()] \
                if res.boxes is not None else []

            _draw_boxes(img, _gt_boxes(test_lbl_dir / f"{stem}.txt"), COLOR_GT, "GT")
            _draw_boxes(img, pred_boxes, COLOR_PRED, "pred")

            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    return fig


def build_summary_text(metrics, results_dir, df_eval):
    """Résumé texte de l'entraînement + métriques niveau image."""
    lines = [f"Rapport d'entraînement YOLO — {RUN_NAME_YOLO}",
             f"Généré : {datetime.now():%Y-%m-%d %H:%M:%S}", "",
             "Métriques (niveau image, comparables au CNN) :"]
    for k, v in metrics.items():
        lines.append(f"  {k:12s} : {v:.4f}")
    lines += ["", f"Images test évaluées : {len(df_eval)}",
              f"  TP={((df_eval.y_true==1)&(df_eval.y_pred==1)).sum()}  "
              f"FN={((df_eval.y_true==1)&(df_eval.y_pred==0)).sum()}  "
              f"TN={((df_eval.y_true==0)&(df_eval.y_pred==0)).sum()}  "
              f"FP={((df_eval.y_true==0)&(df_eval.y_pred==1)).sum()}"]
    return "\n".join(lines)


def save_report_to_gcs(model, metrics, df_eval, results_dir,
                       test_img_dir, test_lbl_dir, imgsz):
    """Génère et pousse sur GCS (suivi_perf/yolo/<run>_<timestamp>/) :
    résumé texte, planche d'exemples, results.csv et results.png du run."""
    results_dir = Path(results_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = f"{PERF_BUCKET_PREFIX_YOLO}/{RUN_NAME_YOLO}_{timestamp}"
    bucket = _bucket()

    def _upload_file(local, name):
        if Path(local).exists():
            bucket.blob(f"{prefix}/{name}").upload_from_filename(str(local))
            return True
        return False

    def _upload_str(text, name):
        bucket.blob(f"{prefix}/{name}").upload_from_string(text, content_type="text/plain")

    # 1. résumé texte
    _upload_str(build_summary_text(metrics, results_dir, df_eval), "summary.txt")

    # 2. planche d'exemples
    fig = build_examples_figure(model, df_eval, test_img_dir, test_lbl_dir, imgsz)
    tmp = results_dir / "predictions_examples.png"
    fig.savefig(tmp, dpi=120, bbox_inches="tight"); plt.close(fig)
    _upload_file(tmp, "predictions_examples.png")

    # 3. courbes natives Ultralytics (déjà générées pendant train)
    _upload_file(results_dir / "results.csv", "results.csv")
    _upload_file(results_dir / "results.png", "results.png")  # courbes train/val

    gs_uri = f"gs://{BUCKET_NAME}/{prefix}/"
    print(f"📊 Rapport YOLO sauvegardé : {gs_uri}")
    return gs_uri
