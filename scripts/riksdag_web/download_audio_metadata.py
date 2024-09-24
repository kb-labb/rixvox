import argparse
import locale
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from rixvox.api import get_audio_metadata
from rixvox.text import preprocess_text

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_dir", type=str, default="data/riksdagen_web")
args = argparser.parse_args()

locale.setlocale(locale.LC_ALL, "sv_SE.UTF-8")  # Swedish date format


def coalesce_columns(df, col1="anftext", col2="anforandetext"):
    """
    Coalesce text columns in df, replacing NaN values (i.e. missing speeches)
    in first column with values from second column.

    Args:
        df (pd.DataFrame): A pandas dataframe with the relevant metadata fields.
        col1 (str): The name of the 1st text column in df whose NaNs we are filling.
        col2 (str): The name of the 2nd text column in df.

    Returns:
        pd.DataFrame: A pandas dataframe with the coalesced text column.
    """

    df[col1] = df[col1].fillna(df[col2])
    df = df.drop(columns=[col2])

    return df


df = pd.read_parquet(os.path.join(args.data_dir, "df_anforanden_metadata.parquet"))
df = df[~pd.isna(df["rel_dok_id"])].reset_index(drop=True)

# Some anforanden have multiple rel_dok_ids, we select the first one
first_rel_dok_id = df[df["rel_dok_id"].str.contains(",")]["rel_dok_id"].str.extract("(.*?)(?=, )")
df.loc[df["rel_dok_id"].str.contains(","), "rel_dok_id"] = first_rel_dok_id.iloc[:, 0].tolist()

# Downlaod audio metadata from unique rel_dok_ids (debates)
with mp.Pool(mp.cpu_count()) as p:
    df_list = p.map(get_audio_metadata, tqdm(df["rel_dok_id"].unique().tolist()), chunksize=20)

df_audiometa = pd.concat(df_list, axis=0)
headers = (
    df.groupby("avsnittsrubrik").size().sort_values(ascending=False).head(10000).index.tolist()
)

with mp.Pool(mp.cpu_count()) as p:
    df_split = p.starmap(
        preprocess_text, [(df, headers) for df in np.array_split(df_audiometa, mp.cpu_count())]
    )

df_audiometa = pd.concat(df_split, axis=0)
df_audiometa = df_audiometa.reset_index(drop=True)
# 2004-04-22 00:00:00 to 2004-04-22
df_audiometa["debatedate"] = pd.to_datetime(df_audiometa["anf_datum"], format="%Y-%m-%d %H:%M:%S")
df_audiometa.loc[df_audiometa["anf_text"] == "", "anf_text"] = None
df_audiometa["anf_nummer"] = df_audiometa["anf_nummer"].astype("int64")

# Some speech texts are missing from audio metadata, we add them from df_anforanden_metadata
df = df.rename(columns={"intressent_id": "intressent_id2_text"})
df["rel_dok_id"] = df["rel_dok_id"].str.upper()
df_audiometa["rel_dok_id"] = df_audiometa["rel_dok_id"].str.upper()

df_audiometa = df_audiometa.merge(
    df[["rel_dok_id", "anforande_nummer", "anforandetext", "talare", "intressent_id2_text"]],
    left_on=["rel_dok_id", "anf_nummer"],
    right_on=["rel_dok_id", "anforande_nummer"],
    how="left",
    suffixes=("_audio", "_text"),
)

# Uppercase all names/party names because they are inconsistent in the data
df_audiometa["parti"] = df_audiometa["parti"].str.upper()
df_audiometa["talare"] = df_audiometa["talare"].str.upper()

# Replace NaN in anf_text column with text from anforandetext where available
df_audiometa = coalesce_columns(df_audiometa, col1="anf_text", col2="anforandetext")

# Drop any duplicates
df_audiometa = df_audiometa[~df_audiometa.duplicated(keep="first")].reset_index(drop=True)
# Drop speeches with no text
df_audiometa = df_audiometa[~df_audiometa["anf_text"].isna()].reset_index(drop=True)

# Remove duplicate name columns
df_audiometa = df_audiometa.loc[:, ~df_audiometa.columns.duplicated("first")]


df_audiometa.to_parquet(os.path.join(args.data_dir, "df_audio_metadata.parquet"), index=False)
