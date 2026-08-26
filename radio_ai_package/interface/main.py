# Imports du main
import pandas as pd
import numpy as np
import os
from pathlib import Path
from radio_ai_package.params import *
import sys


# Imports de Luca
from radio_ai_package.ml_logic.data import load_df_with_local_paths, import_data_bucket










# Imports de Modibo / Merwan






# Imports de Mariana






#### zone de Luca (upload data)

if Path(BASE_DIR).name != "radio_ai" : sys.exit("ATTENTION : erreur de localisation radio_ia")

# df = load_df_with_local_paths()
df = import_data_bucket()
print(df.file_path.sample(1).str[-40:])











































### zone de Modibo / Merwan (preprocess et modele)



































### zone de Mariana (perf du modele)
