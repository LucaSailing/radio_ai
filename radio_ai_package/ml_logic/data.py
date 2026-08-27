import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
from radio_ai_package.params import RAW_DATA_DIR, IMAGE_DIRS, BASE_DIR, BUCKET_NAME


def _build_local_index():
    """Index {nom_fichier.png -> chemin absolu} construit en un seul balayage
    des dossiers d'images. Réutilisé pour renseigner 'file_path' en O(n)."""
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

    # affectation vectorisée (O(n)) : on mappe filestem+'.png' sur l'index.
    # Remplace la boucle O(n²) qui rebalayait le df à chaque fichier.
    df["file_path"] = (df["filestem"] + ".png").map(index).fillna("")

    # petit reporting : combien d'images ont bien été trouvées en local
    trouvees = (df["file_path"] != "").sum()
    print(f"  file_path renseigné : {trouvees}/{len(df)} images trouvées en local")

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
                                   "folder_structure"),
                       max_workers=16):
    """Télécharge dataset + images (skip ce qui est déjà local), puis renseigne
    'file_path'. Downloads parallélisés (I/O bound)."""
    storage_client = storage.Client()

    # 1. dataset
    import_data_file("raw_data/dataset.csv", storage_client)
    df = pd.read_csv(RAW_DATA_DIR / "dataset.csv")

    # 2. liste des blobs à récupérer, filtrée sur les parties voulues
    to_download = [b.name for b in storage_client.list_blobs(BUCKET_NAME)
                   if any(s in b.name for s in image_sets)]
    print(f"  {len(to_download)} fichiers ciblés dans le bucket")

    # 3. downloads en parallèle — chaque tâche skippe si déjà local
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pool.map(lambda n: import_data_file(n, storage_client), to_download)

    # 4. chemins locaux renseignés en vectoriel
    return load_df_with_local_paths(df)
