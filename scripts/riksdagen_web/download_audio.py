import argparse
import logging
import multiprocessing as mp
import os

import pandas as pd
from tqdm import tqdm

from rixvox.api import audio_to_dokid_folder, audiofile_exists, get_media_file

logging.basicConfig(
    filename="logs/download_audio.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--audio_dir",
    type=str,
    default="data/audio/2000_2024",
    help="Path to directory with audio files.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="data/riksdagen_web",
    help="Path to directory with data.",
)
args = parser.parse_args()


# Read data
df = pd.read_parquet(os.path.join(args.data_dir, "df_audio_metadata.parquet"))


# Subset only speeches which we haven't already downloaded audio for
df["audiofile_exists"] = df[["dok_id", "audiofileurl"]].apply(
    lambda x: audiofile_exists(
        x.dok_id, x.audiofileurl.rsplit("/", 1)[1], audio_dir=args.audio_dir
    ),
    axis=1,
)


df_undownloaded = df[df["audiofile_exists"] == False].reset_index(drop=True)
audio_download_urls = df_undownloaded["audiofileurl"].unique().tolist()

# Download audio files
logger.info("Downloading audio files")
with mp.Pool(mp.cpu_count()) as p:
    p.starmap(
        get_media_file,
        tqdm(
            # url, backoff_factor, pbar (per file), audio_dir
            [(audiofileurl, 0.2, False, args.audio_dir) for audiofileurl in audio_download_urls],
            total=len(audio_download_urls),
        ),
        chunksize=5,
    )


# Move audio files to folder named after dokid
logger.info("Moving audio files to dokid folders")
audio_to_dokid_folder(df, audio_dir=args.audio_dir)


# Add inferred filepaths to df (some files may not have been downloaded)
df["filename"] = df.apply(
    lambda x: x["dok_id"] + "/" + x["audiofileurl"].rsplit("/", 1)[1], axis=1
)
df = df.drop(columns=["audiofile_exists"])

# Check the filesize of every filename in KBytes
df["filesize"] = df["filename"].apply(
    lambda x: (
        os.path.getsize(os.path.join(args.audio_dir, x)) / 1024
        if os.path.exists(os.path.join(args.audio_dir, x))
        else None
    )
)

# Remove files < 70KB
df = df[~pd.isna(df["filesize"]) & (df["filesize"] > 70)].reset_index(drop=True)
df = df.drop(columns=["filesize"])
df["duration"] = df["duration"].astype("Int64")
df = df[df["filename"] != "GZ1035/2442208010021962521_aud.mp3"].reset_index(
    drop=True
)  # corrupted file

logging.info(
    f"Total duration of speeches according to the Riksdag's metadata: {df['duration'].sum() / 60 / 60} hours"
)

# Save updated df with audio filepaths
df.to_parquet(os.path.join(args.data_dir, "df_audio_metadata.parquet"), index=False)
