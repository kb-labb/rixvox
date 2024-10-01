import argparse
import glob
import logging
import multiprocessing as mp
import os
from itertools import repeat
from pathlib import Path

import pandas as pd
import simplejson as json
from tqdm import tqdm

from rixvox.audio import convert_audio_to_wav
from rixvox.dataset import read_json, read_json_parallel
from rixvox.json import preprocess_json
from rixvox.speech_finder import contiguous_fuzzy_match_split
from rixvox.text import normalize_text

logging.basicConfig(
    filename="logs/string_match.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--json_dir",
    type=str,
    default="data/vad_wav2vec_output",
)
argparser.add_argument(
    "--data_dir",
    type=str,
    default="data/riksdagen_web",
)
argparser.add_argument(
    "--num_workers",
    type=int,
    default=16,
)
args = argparser.parse_args()


# Get the start_time and end_time from text_timestamps for word_start and word_end
def get_timestamps(row):
    """
    # Get the start_time and end_time from text_timestamps for word_start and word_end
    """
    # If both are NAType, return None, None
    if pd.isna(row["word_start"]) and pd.isna(row["word_end"]):
        return None, None
    elif pd.isna(row["word_start"]):
        return None, row["text_timestamps"][row["word_end"] - 1]["end_time"]
    elif pd.isna(row["word_end"]):
        return row["text_timestamps"][row["word_start"]]["start_time"], None

    try:
        return (
            row["text_timestamps"][row["word_start"]]["start_time"],
            row["text_timestamps"][row["word_end"] - 1]["end_time"],
        )
    except IndexError:
        logger.error(
            f"IndexError: {row['word_start']}, {row['word_end']}. Total words: {len(row['text_timestamps'])}"
        )

        if len(row["text_timestamps"]) < (row["word_end"] - 1):
            # In rare cases, the word_end is slightly larger than the length of text_timestamps.
            # No time to debug what is causing this, so just return the last timestamp
            return (
                row["text_timestamps"][row["word_start"]]["start_time"],
                row["text_timestamps"][-1]["end_time"],
            )

        return -100, -100


df = pd.read_parquet("data/riksdagen_web/df_audio_metadata.parquet")

with mp.Pool(args.num_workers) as pool:
    df["text_normalized"] = list(
        tqdm(
            pool.imap(normalize_text, df["anf_text"].tolist()),
            total=len(df),
            desc="Normalizing text",
        )
    )

json_files = glob.glob(f"{args.json_dir}/*.json")

# Chunk json files into batches of 100
json_files = [json_files[i : i + 100] for i in range(0, len(json_files), 100)]
with mp.Pool(args.num_workers + 4) as pool:
    df_audio = pool.map(preprocess_json, tqdm(json_files, total=len(json_files)), chunksize=1)

df_audio = pd.concat(df_audio, ignore_index=True)  # inference from audio

df_audio["filename"] = df_audio["audio_path"].apply(lambda x: os.path.join(*Path(x).parts[-2:]))
df = df.merge(
    df_audio[["audio_path", "text_timestamps", "inference_normalized", "filename"]],
    how="left",
    left_on="filename",
    right_on="filename",
)

# How many inference_normalized are missing?
df = df[~df["inference_normalized"].isna()].reset_index(drop=True)

# Fuzzy string match the text_normalized (speech) and inference_normalized (audio file)
with mp.Pool(args.num_workers + 3) as pool:
    args_fuzzy = zip(
        df["text_normalized"].tolist(),
        df["inference_normalized"].tolist(),
        repeat(55),  # threshold, see contiguous_fuzzy_match_split for details
        repeat(120),  # max_length
    )
    scores = pool.starmap(
        contiguous_fuzzy_match_split,
        tqdm(args_fuzzy, total=len(df)),
        chunksize=100,
    )


df[["word_start", "word_end", "score"]] = pd.DataFrame(scores)

# Make word_start and word_end integers and allow for None/NaN
df[["word_start", "word_end"]] = df[["word_start", "word_end"]].astype("Int64")

df[["start_time", "end_time"]] = df[["text_timestamps", "word_start", "word_end"]].apply(
    get_timestamps, axis=1, result_type="expand"
)

# Remove speeches where both word_start and word_end are NA
df_aligned = df[~(df["word_start"].isna() & df["word_end"].isna())].reset_index(drop=True)

df_aligned = df_aligned.drop(["inference_normalized", "text_timestamps"], axis=1)
df_aligned["duration_text"] = df_aligned["end_time"] - df_aligned["start_time"]
df_aligned = df_aligned[~df_aligned["duration_text"].isna()].reset_index(drop=True)

df_aligned.to_parquet(os.path.join(args.data_dir, "string_aligned_speeches.parquet"), index=False)
