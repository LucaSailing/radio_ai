"""
preprocessor_yolo.py — préparation du dataset au format YOLO (radio_ai)

Réutilise le split par patient de preprocessor_CNN (mêmes proportions
train/val/test que le CNN, pour permettre une comparaison directe des deux
modèles) et construit l'arborescence images/labels attendue par Ultralytics.
Les images sont recherchées directement dans images_part1 à 4 (avec
cnn_images en secours), donc pas besoin d'une étape de copie préalable.
"""

import shutil
from pathlib import Path

from radio_ai_package.params import RAW_DATA_DIR, TEST_SIZE, VAL_SIZE, RANDOM_STATE
from radio_ai_package.ml_logic.preprocessors.preprocessor_CNN import filtering

# Classe "fracture" dans les annotations brutes GRAZPEDWRI-DX (0=boneanomaly,
# 1=bonelesion, 2=foreignbody, 3=fracture, 4=metal, 5=periostealreaction,
# 6=pronatorsign, 7=softtissue, 8=text) — remappée en classe 0 pour YOLO
# puisqu'on n'entraîne que sur "fracture".
FRACTURE_CLASS_ID = "3"

# Dossiers où chercher les images, dans l'ordre de priorité
IMAGE_SOURCE_DIRS = [
    RAW_DATA_DIR / "images_part1",
    RAW_DATA_DIR / "images_part2",
    RAW_DATA_DIR / "images_part3",
    RAW_DATA_DIR / "images_part4",
    RAW_DATA_DIR / "cnn_images",
]

YOLO_LABELS_SRC_DEFAULT = RAW_DATA_DIR / "folder_structure" / "yolov5" / "labels"
YOLO_DATASET_DIR_DEFAULT = RAW_DATA_DIR / "yolo_dataset"


def resolve_image_path(filestem: str) -> Path | None:
    """Cherche l'image dans images_part1-4 puis cnn_images, retourne le premier chemin trouvé."""
    for source_dir in IMAGE_SOURCE_DIRS:
        candidate = source_dir / f"{filestem}.png"
        if candidate.exists():
            return candidate
    return None


def remap_label_file(src_path: Path, dst_path: Path) -> None:
    """Ne garde que les lignes 'fracture' d'un .txt YOLO et les réécrit en classe 0."""
    lines_out = []
    if src_path.exists():
        for line in src_path.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == FRACTURE_CLASS_ID:
                lines_out.append(" ".join(["0"] + parts[1:]))
    dst_path.write_text("\n".join(lines_out))


def _split_by_patient(data_model, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE):
    """Split train/val/test par patient (évite les fuites) — identique à preprocessor_CNN."""
    patients = data_model["patient_id"].drop_duplicates().sample(
        frac=1, random_state=random_state
    ).tolist()
    n = len(patients)
    i_train = round(n * (1 - test_size - val_size))
    i_val = round(n * (1 - test_size))
    train_ids = set(patients[:i_train])
    val_ids = set(patients[i_train:i_val])
    test_ids = set(patients[i_val:])
    return (
        data_model[data_model["patient_id"].isin(train_ids)],
        data_model[data_model["patient_id"].isin(val_ids)],
        data_model[data_model["patient_id"].isin(test_ids)],
    )


def _populate_split(df_split, yolo_labels_src, dataset_dir, split_name):
    """Copie les images (résolues via IMAGE_SOURCE_DIRS) et les labels remappés d'un split."""
    img_dir = dataset_dir / "images" / split_name
    lbl_dir = dataset_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for stem in df_split["filestem"]:
        src_img = resolve_image_path(stem)
        if src_img is None:
            missing.append(stem)
            continue
        shutil.copy(src_img, img_dir / f"{stem}.png")
        remap_label_file(yolo_labels_src / f"{stem}.txt", lbl_dir / f"{stem}.txt")

    if missing:
        print(f"⚠️  {len(missing)} images introuvables pour le split '{split_name}' (ex: {missing[:3]})")


def build_yolo_dataset(
    df,
    yolo_labels_src: Path = YOLO_LABELS_SRC_DEFAULT,
    dataset_dir: Path = YOLO_DATASET_DIR_DEFAULT,
):
    """
    Filtre df, split par patient (train/val/test), copie images+labels remappés
    dans l'arborescence Ultralytics, écrit fracture_dataset.yaml.

    Retourne : (yaml_path, dataset_dir), (data_train, data_val, data_test)
    """
    df_filtered = filtering(df)
    data_train, data_val, data_test = _split_by_patient(df_filtered)

    _populate_split(data_train, yolo_labels_src, dataset_dir, "train")
    _populate_split(data_val, yolo_labels_src, dataset_dir, "val")
    _populate_split(data_test, yolo_labels_src, dataset_dir, "test")

    yaml_path = dataset_dir / "fracture_dataset.yaml"
    yaml_path.write_text(
        f"path: {dataset_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        "  0: fracture\n"
    )

    print(f"✅ Dataset YOLO prêt : {len(data_train)} train / {len(data_val)} val / {len(data_test)} test")
    return (yaml_path, dataset_dir), (data_train, data_val, data_test)
