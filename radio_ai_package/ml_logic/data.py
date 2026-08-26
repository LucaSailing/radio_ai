import glob
import pandas as pd
from pathlib import Path
from google.cloud import storage
from radio_ai_package.params import RAW_DATA_DIR, IMAGE_DIRS, BASE_DIR, BUCKET_NAME

def load_df_with_local_paths(df=None):
    """Renseigne 'file_path' SANS toucher au bucket : suppose les images déjà
    présentes dans raw_data/images_partX. Voie rapide au quotidien."""
    if df is None:
        df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")

    # index {nom_fichier.png -> chemin absolu} en un seul balayage disque
    index = {}
    for d in IMAGE_DIRS:
        for path in Path(RAW_DATA_DIR  / d).glob("*.png"):
            index[path.name] = str(path)

    for stem in df["filestem"]:
        df.loc[df.filestem == stem, "file_path"] = index.get(stem + ".png", "")

    return df


def import_data_file(storage_filename, storage_client):
    """Télécharge un fichier du bucket vers un chemin local répliquant
    l'arborescence en ligne. Skip si déjà présent."""
    local_filename = BASE_DIR / storage_filename
    if local_filename.exists():
        return
    local_filename.parent.mkdir(parents=True, exist_ok=True)
    bucket = storage_client.bucket(BUCKET_NAME)
    bucket.blob(storage_filename).download_to_filename(str(local_filename))


def import_data_bucket(image_sets=("images_part1", "images_part2",
                                   "images_part3", "images_part4",
                                   "folder_structure")):
    """Télécharge dataset + images (skip ce qui est déjà local), puis renseigne
    'file_path'."""
    storage_client = storage.Client()

    import_data_file("raw_data/dataset.csv", storage_client)
    df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")

    to_download = [b.name for b in storage_client.list_blobs(BUCKET_NAME)
                   if any(s in b.name for s in image_sets)]

    for name in to_download:
        import_data_file(name, storage_client)

    return load_df_with_local_paths(df)
