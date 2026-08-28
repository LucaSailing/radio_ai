# params Luca
from pathlib import Path

BUCKET_NAME = "radio-ai_bucket"
PROJECT_ID = "radio-ai-506510"
IMAGE_DIRS = ["images_part1", "images_part2", "images_part3", "images_part4"]
ROOT_NAME = "radio_ai"

# --- chemins LOCAUX (sur le disque) -----------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"

# --- préfixes dans le BUCKET GCS --------------------------------------------
BUCKET_NAME = "radio-ai_bucket"
RAW_DATA_BUCKET_PREFIX = "raw_data"    # gs://<bucket>/raw_data/
MODEL_BUCKET_PREFIX = "models"         # gs://<bucket>/models/

# paramètres de preprocessing
#IMG_SIZE = (256, 256)
IMG_SIZE = (384, 384)
#IMG_SIZE = (512, 512)
BATCH_SIZE = 16                 # renommé (était BATCH)
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# paramètres de modèle
LEARNING_RATE = 1e-3            # ajouté (requis par model_CNN)
EPOCHS = 40
PATIENCE = 10                    # abaissé pour être < EPOCHS


# --- mode de chargement des images ------------------------------------------
DATA_MODE = "local"        # "local" ou "remote"

# colonne de chemins correspondante, dérivée automatiquement du mode
PATH_COL = "file_path_gs" if DATA_MODE == "remote" else "file_path"


# --- mode d'exécution -------------------------------------------------------
RUN_MODE = "train"    # "train" = entraîne puis évalue | "eval" = charge un modèle existant et évalue
MODEL_TO_LOAD = None  # nom du modèle à charger en mode "eval" (ex. "cnn_fracture_20260827-160541.keras")
                      # None = prend le plus récent dans le bucket
