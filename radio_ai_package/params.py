# params Luca
from pathlib import Path

BUCKET_NAME = "radio-ai_bucket"
PROJECT_ID = "radio-ai-506510"
IMAGE_DIRS = ["images_part1", "images_part2", "images_part3", "images_part4"]
ROOT_NAME = "radio_ai"
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "raw_data"
MODEL_DIR = RAW_DATA_DIR / "models"

# paramètres de preprocessing
IMG_SIZE = (224, 224)
#IMG_SIZE = (528, 528)
BATCH_SIZE = 32                 # renommé (était BATCH)
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42

# paramètres de modèle
LEARNING_RATE = 1e-3            # ajouté (requis par model_CNN)
EPOCHS = 6
PATIENCE = 3                    # abaissé pour être < EPOCHS
