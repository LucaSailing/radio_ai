import io
import os
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
from radio_ai_package.params import (RAW_DATA_DIR, IMAGE_DIRS, BASE_DIR,
                                     BUCKET_NAME, DATA_MODE)


# ============================================================================
#  Point d'entrée unique — bascule local / remote selon DATA_MODE (params.py)
# ============================================================================

def load_data():
    """Charge le dataset et renseigne les chemins d'images selon DATA_MODE :
    - 'local'  : images déjà dans raw_data/ -> colonne 'file_path'
    - 'remote' : URI gs:// vers le bucket   -> colonne 'file_path_gs'
    Pour switcher, changer DATA_MODE dans params.py (rien d'autre à toucher)."""
    if DATA_MODE == "remote":
        return load_df_with_remote_paths()
    elif DATA_MODE == "local":
        return load_df_with_local_paths()
    else:
        raise ValueError(f"DATA_MODE inconnu : {DATA_MODE!r} (attendu 'local' ou 'remote')")


# ============================================================================
#  Mode LOCAL — lecture des images déjà présentes dans raw_data/
# ============================================================================

def _build_local_index():
    """Index {nom_fichier.png -> chemin absolu} construit en un seul balayage
    des dossiers d'images. Permet de renseigner 'file_path' en O(n)."""
    index = {}
    for d in IMAGE_DIRS:
        for path in (RAW_DATA_DIR / d).glob("*.png"):
            index[path.name] = str(path)
    return index


def load_df_with_local_paths(df=None):
    """Renseigne 'file_path' SANS toucher au bucket : suppose les images déjà
    présentes dans raw_data/images_partX. Voie rapide au quotidien."""
    if df is None:
        df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")

    index = _build_local_index()
    df["file_path"] = (df["filestem"] + ".png").map(index).fillna("")

    trouvees = (df["file_path"] != "").sum()
    print(f"  file_path renseigné : {trouvees}/{len(df)} images trouvées en local")
    return df


# ============================================================================
#  Téléchargement du bucket vers le disque local (peuplement initial)
# ============================================================================

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
                                   "folder_structure"),
                       max_workers=16):
    """Télécharge dataset + images (skip ce qui est déjà local), puis renseigne
    'file_path'. Downloads parallélisés (I/O bound). À utiliser pour peupler
    une machine neuve, avant de travailler en mode 'local'."""
    storage_client = storage.Client()

    import_data_file("raw_data/dataset.csv", storage_client)
    df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")

    to_download = [b.name for b in storage_client.list_blobs(BUCKET_NAME)
                   if any(s in b.name for s in image_sets)]
    print(f"  {len(to_download)} fichiers ciblés dans le bucket")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pool.map(lambda n: import_data_file(n, storage_client), to_download)

    return load_df_with_local_paths(df)


# ============================================================================
#  Mode REMOTE — chemins gs:// lus nativement par tf.io.read_file
# ============================================================================

def load_df_with_remote_paths(df=None):
    """Renseigne 'file_path_gs' avec des URI gs:// pointant vers le bucket
    (pas de copie locale). Pensé pour l'entraînement sur VM éphémère :
    tf.io.read_file lit nativement les chemins gs://."""
    storage_client = storage.Client()

    if df is None:
        blob = storage_client.bucket(BUCKET_NAME).blob("raw_data/dataset.csv")
        df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))

    # {nom_fichier.png -> chemin complet dans le bucket}
    path_by_basename = {
        os.path.basename(blob.name): blob.name
        for blob in storage_client.list_blobs(BUCKET_NAME)}

    gs_prefix = f"gs://{BUCKET_NAME}/"
    df['file_path_gs'] = (df['filestem'] + '.png').map(path_by_basename)
    df['file_path_gs'] = df['file_path_gs'].map(
        lambda name: gs_prefix + name if pd.notna(name) else "")

    trouvees = (df['file_path_gs'] != "").sum()
    print(f"  file_path_gs (gs://) renseigné : {trouvees}/{len(df)} images trouvées dans le bucket")
    return df
