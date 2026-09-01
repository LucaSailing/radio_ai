# params.py

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

BUCKET_NAME = os.environ.get("BUCKET_NAME")
PROJECT_ID = os.environ.get("PROJECT_ID")

IMAGE_DIRS = ["images_part1", "images_part2", "images_part3", "images_part4"]
ROOT_NAME = "radio_ai"

# --- chemins LOCAUX ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"

# --- préfixes BUCKET GCS ----------------------------------------------------
RAW_DATA_BUCKET_PREFIX = "raw_data"
MODEL_BUCKET_PREFIX = "models"
MODEL_BUCKET_PREFIX_YOLO = "models/yolo"
CKPT_BUCKET_PREFIX_YOLO = os.environ.get("CKPT_BUCKET_PREFIX_YOLO", "checkpoints/yolo")

# --- preprocessing (castés : tout sort du .env en string) -------------------
IMG_SIZE = ( int(os.environ.get("IMG_SIZE_WIDTH")),
             int(os.environ.get("IMG_SIZE_HEIGHT")) )
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
TEST_SIZE = float(os.environ.get("TEST_SIZE"))
VAL_SIZE = float(os.environ.get("VAL_SIZE"))

RANDOM_STATE = 42

# --- modèle -----------------------------------------------------------------
LEARNING_RATE = float(os.environ.get("LEARNING_RATE"))
EPOCHS = int(os.environ.get("EPOCHS"))
PATIENCE = int(os.environ.get("PATIENCE"))

# --- mode de chargement des images ------------------------------------------
DATA_MODE = os.environ.get("DATA_MODE")
PATH_COL = "file_path_gs" if DATA_MODE == "remote" else "file_path"

# --- mode d'exécution -------------------------------------------------------
RUN_MODE = os.environ.get("RUN_MODE")

_model = os.environ.get("MODEL_TO_LOAD", "None")
MODEL_TO_LOAD = None if _model in ("None", "") else _model

# --- mode evaluation  -------------------------------------------------------
THRESHOLD = float(os.environ.get("THRESHOLD"))



# =========================================================================
# YOLO PARAMETERS
# =========================================================================

# --- hyperparamètres d'entraînement ---
EPOCH_YOLO      = int(os.environ.get("EPOCH_YOLO", "50"))
IMAGE_SIZE_YOLO = int(os.environ.get("IMAGE_SIZE_YOLO", "256"))
PATIENCE_YOLO   = int(os.environ.get("PATIENCE_YOLO", "10"))
BATCH_SIZE_YOLO = int(os.environ.get("BATCH_SIZE_YOLO", str(BATCH_SIZE)))  # défaut = batch CNN

# --- poids de départ ---
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")

# --- runtime (device / workers) ---
DEVICE_YOLO  = os.environ.get("DEVICE_YOLO", "auto")      # "auto" | "cpu" | "cuda" | "0"
WORKERS_YOLO = int(os.environ.get("WORKERS_YOLO", "0"))   # 0 sur CPU, >0 sur GPU

# --- sauvegarde / reprise ---
SAVE_PERIOD_YOLO = int(os.environ.get("SAVE_PERIOD_YOLO", "5"))       # checkpoint tous les N epochs
RUN_NAME_YOLO    = os.environ.get("RUN_NAME_YOLO", "fracture_yolov8n")  # <-- MANQUAIT : cause du crash
