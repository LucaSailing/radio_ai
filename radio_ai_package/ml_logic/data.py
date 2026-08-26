import glob
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

BUCKET_NAME = "radio-ai_bucket"
PROJECT_ID = "radio-ai-506510"
IMAGE_DIRS = ["images_part1", "images_part2", "images_part3", "images_part4"]


def _find_root(marker="radio_ai"):
    """Remonte depuis ce fichier jusqu'au dossier racine du package.
    La racine est garantie d'exister ; sinon on casse avec une erreur explicite."""
    for d in Path(__file__).resolve().parents:
        if d.name == marker:
            return d
    raise FileNotFoundError(f"Racine '{marker}' introuvable dans l'arborescence.")


BASE_DIR = _find_root()
RAW_DATA_DIR = BASE_DIR / "raw_data"


def add_local_paths(df=None):
    """Renseigne 'file_path' SANS toucher au bucket : suppose les images déjà
    présentes dans raw_data/images_partX. Voie rapide au quotidien."""
    if df is None:
        df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")

    # index {nom_fichier.png -> chemin absolu} en un seul balayage disque
    index = {}
    for d in IMAGE_DIRS:
        for path in (RAW_DATA_DIR / d).glob("*.png"):
            index[path.name] = str(path)

    # --- ACCÉLÉRATION (affectation) -----------------------------------------
    # Version simple, lisible mais O(n²) : une passe sur df par fichier.
    for stem in df["filestem"]:
        df.loc[df.filestem == stem, "file_path"] = index.get(stem + ".png", "")

    # Version rapide, O(n) : dict + map vectorisé. Décommenter pour activer
    # (et commenter la boucle ci-dessus). C'est LE gros gain quand les
    # fichiers sont déjà en local.
    # df["file_path"] = (df["filestem"] + ".png").map(index).fillna("")
    # ------------------------------------------------------------------------
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

    # --- ACCÉLÉRATION (listing du bucket) -----------------------------------
    # Version simple : un seul listing global du bucket, filtré côté client.
    to_download = [b.name for b in storage_client.list_blobs(BUCKET_NAME)
                   if any(s in b.name for s in image_sets)]

    # Version rapide : un listing par préfixe (moins de données transférées,
    # surtout si le bucket contient d'autres choses). Décommenter pour activer
    # (et commenter le bloc ci-dessus).
    # to_download = []
    # for s in image_sets:
    #     to_download += [b.name for b in
    #                     storage_client.list_blobs(BUCKET_NAME, prefix=f"raw_data/{s}")]
    # ------------------------------------------------------------------------

    # --- ACCÉLÉRATION (downloads) -------------------------------------------
    # Version simple : téléchargements séquentiels.
    for name in to_download:
        import_data_file(name, storage_client)

    # Version rapide : downloads parallèles (I/O bound → gain quasi linéaire).
    # N'aide qu'au PREMIER téléchargement ; inutile si tout est déjà local.
    # Décommenter pour activer (et commenter la boucle ci-dessus).
    # with ThreadPoolExecutor(max_workers=16) as pool:
    #     list(pool.map(lambda n: import_data_file(n, storage_client), to_download))
    # ------------------------------------------------------------------------

    return add_local_paths(df)
