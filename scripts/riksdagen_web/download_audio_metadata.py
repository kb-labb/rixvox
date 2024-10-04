import argparse
import locale
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from numpy import nanmax, nanmin
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


df_audiometa = pd.read_parquet(os.path.join(args.data_dir, "df_audio_metadata.parquet"))

#### Add speaker metadata to audio metadata ####
df_person = pd.read_csv(args.data_dir + "/person.csv")

df_person["From"] = pd.to_datetime(df_person["From"], format="ISO8601")
df_person["Tom"] = pd.to_datetime(df_person["Tom"], format="ISO8601")

# If a person has been in several parties, we concatenate them (not repeating the same party)
party_affiliation = (
    df_person.groupby("Id")
    .agg(
        {
            "From": lambda x: nanmin(x) if not x.isnull().all() else pd.NaT,
            "Tom": lambda x: nanmax(x) if not x.isnull().all() else pd.NaT,
            "Parti": lambda x: ", ".join(set(x.dropna())),
            "Uppdragsroll": lambda x: ", ".join(set(x.dropna())),
        }
    )
    .reset_index()
)


df_person = df_person.groupby("Id").first().reset_index()

party_affiliation = party_affiliation.rename(columns={"Parti": "party", "Uppdragsroll": "role"})
df_person = df_person.merge(party_affiliation, on="Id", how="left", suffixes=("", "_party"))

df_person["name"] = df_person["Förnamn"] + " " + df_person["Efternamn"]
df_person["born"] = pd.to_datetime(df_person["Född"], format="ISO8601")
df_person["gender"] = df_person["Kön"]
# kvinna = woman in gender columns
df_person.loc[df_person["gender"] == "kvinna", "gender"] = "woman"
df_person["district"] = df_person["Valkrets"]
df_person = df_person.rename(columns={"Id": "riksdagen_id"})


df = df_audiometa.merge(
    df_person[["riksdagen_id", "name", "party", "gender", "born", "district", "role"]],
    left_on="intressent_id2",
    right_on="riksdagen_id",
    how="left",
)
df["speaker_id"] = None

# Remove titles and party abbrevations to make it as similar as possible in format to df["speaker"]
# Temporary column to be able to join names from "text" column to "speaker" column in a common format.
df["name_temp"] = df["talare"].str.lower()
df["name_temp"] = df["name_temp"].str.replace(".+?minister", "", regex=True)
df["name_temp"] = df["name_temp"].str.replace(".+?min\.", "", regex=True)
df["name_temp"] = df["name_temp"].str.replace(".+?rådet", "", regex=True)
df["name_temp"] = df["name_temp"].str.replace(".+?[T|t]alman", "", regex=True)
df["name_temp"] = df["name_temp"].str.replace("Talman", "")
# Remove everything within parenthesis
df["name_temp"] = df["name_temp"].str.replace("\(.*\)", "", regex=True)
df["name_temp"] = df["name_temp"].str.strip()
# Change first letter of each part of name to Capitalized
df["name_temp"] = df["name_temp"].apply(lambda x: x.title() if x is not None else None)

df["speaker_from_id"] = True
df.loc[df["name_temp"].isna(), "speaker_from_id"] = False
df = coalesce_columns(df, col1="name", col2="name_temp")
df = coalesce_columns(df, col1="party", col2="parti")
df = coalesce_columns(df, col1="riksdagen_id", col2="intressent_id2")

df["party"] = df["party"].str.upper()
df["speech_id"] = df["dok_id"] + "-" + df["anf_nummer"].astype("str")

df.to_parquet(os.path.join(args.data_dir, "df_audio_metadata.parquet"), index=False)
