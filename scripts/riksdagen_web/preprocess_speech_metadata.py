import argparse
import glob
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from rixvox.text import preprocess_text

parser = argparse.ArgumentParser(
    description="""Read json files of riksdagens anföranden, save relevant metadata fields to file."""
)
parser.add_argument("-f", "--json_dir", type=str, default="/data/faton/riksdagen_old/json")
parser.add_argument("-d", "--out_dir", type=str, default="data/riksdagen_web")
args = parser.parse_args()

json_files = glob.glob(os.path.join(args.json_dir, "*/*.json"))

json_speeches = []

for file in tqdm(json_files):
    with open(os.path.join(file), "r", encoding="utf-8-sig") as f:
        json_speeches.append(json.load(f)["anforande"])

print("Normalizing json to dataframe...")
df = pd.json_normalize(json_speeches)
# df = df.drop(columns=["anforandetext"])
df["anforande_nummer"] = df["anforande_nummer"].astype(int)

# Headers to clean up when next script is run (download_audio_metadata.py)
headers = (
    df.groupby("avsnittsrubrik").size().sort_values(ascending=False).head(10000).index.tolist()
)

# Split df into parts so we can preprocess in parallel
df_list = np.array_split(df, 16)

print("Preprocessing text...")
with mp.Pool(16) as pool:
    df_list = pool.starmap(preprocess_text, [(df, headers, "anforandetext") for df in df_list])

df = preprocess_text(df, headers=headers, textcol="anforandetext")

df = df.sort_values(["dok_id", "anforande_nummer"]).reset_index(drop=True)
df.loc[df["rel_dok_id"] == "", "rel_dok_id"] = None

os.makedirs(args.out_dir, exist_ok=True)
print(f"Saving file to {os.path.join(args.out_dir, 'df_anforanden_metadata.parquet')}")
df.to_parquet(os.path.join(args.out_dir, "df_anforanden_metadata.parquet"), index=False)
