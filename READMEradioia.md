# Radio AI — détection de fractures sur radiographies

Projet de fin de bootcamp **Data Science & IA — Le Wagon** (Paris, 2026), réalisé en équipe.

Radio AI analyse une radiographie et indique s'il y a une fracture, **où** elle se trouve, et
**pourquoi** le modèle le pense — pour aider au tri en imagerie médicale.

## Approche

| Tâche | Modèle | Rôle |
|---|---|---|
| Classification fracture / pas de fracture | CNN maison et VGG (transfer learning) | Premier tri |
| Localisation | YOLO (Ultralytics) | Boîtes autour des fractures, IoU vs vérité terrain |
| Explicabilité | Grad-CAM | Carte de chaleur des zones qui ont pesé dans la décision |

Gestion du déséquilibre de classes (class weights), early stopping, checkpoints synchronisés
sur Google Cloud Storage, rapport de performance par modèle.

## Architecture

```
radio_ai_package/
  ml_logic/
    data.py              chargement des images (local ou bucket GCS)
    preprocessors/       préparation CNN, VGG, YOLO
    models/              model_CNN, model_vgg, model_yolo
    grad_cam.py          explicabilité
    performance_metrics.py, reporting/
    registry.py          sauvegarde / chargement des modèles
  interface/             pipelines d'entraînement (main, main_vgg, main_yolo)
  api/fast.py            API FastAPI — endpoint /analyze
notebooks_modibo/        nettoyage et préparation du jeu de données
```

## Lancer

```bash
pip install -r requirements.txt     # TensorFlow adapté au processeur (Mac / Linux)
# créer un .env :                   BUCKET_NAME, PROJECT_ID, tailles d'image, hyperparamètres
make run                            # entraînement CNN
uvicorn radio_ai_package.api.fast:app --reload   # API
```

## Stack

Python · TensorFlow / Keras · Ultralytics YOLO · OpenCV · scikit-learn · FastAPI ·
Streamlit · Docker · Google Cloud Storage

## Équipe

Projet collectif Le Wagon — dont **Modibo Magassa** (préparation et nettoyage des données,
voir `notebooks_modibo/`).
