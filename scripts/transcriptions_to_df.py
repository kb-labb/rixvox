import argparse
import glob
import logging
import multiprocessing as mp
import sqlite3
from itertools import repeat

import pandas as pd
from tqdm import tqdm

from rixvox.dataset import read_json, read_json_parallel
from rixvox.speech_finder import contiguous_fuzzy_match_split
from rixvox.text import normalize_text

logging.basicConfig(
    filename="logs/string_match.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_global_timestamps(text_timestamps, start):
    for timestamp in text_timestamps:
        timestamp["start_time"] += start / 1000
        timestamp["end_time"] += start / 1000
    return text_timestamps


def preprocess_transcriptions(transcriptions):
    # Expand nested list[dict] column "transcription" to columns
    df_transcription = pd.json_normalize(transcriptions, "chunks", ["metadata"])
    df_transcription["text_timestamps"] = df_transcription["transcription"].apply(
        lambda x: x[0]["word_timestamps"]
    )
    df_transcription["text"] = df_transcription["transcription"].apply(lambda x: x[0]["text"])
    df_transcription["model"] = df_transcription["transcription"].apply(lambda x: x[0]["model"])
    df_transcription = df_transcription.drop(columns=["transcription"])
    df_transcription["dates"] = df_transcription["metadata"].apply(lambda x: x["dates"])
    df_transcription["audio_path"] = df_transcription["metadata"].apply(lambda x: x["audio_path"])
    df_transcription = df_transcription.drop(columns=["metadata"])
    df_transcription["date_start"] = df_transcription["dates"].apply(lambda x: x[0])
    df_transcription["date_end"] = df_transcription["dates"].apply(lambda x: x[-1])
    df_transcription["inference_normalized"] = df_transcription["text"].apply(normalize_text)
    df_transcription["duration"] = df_transcription["end"] - df_transcription["start"]

    # Add start to each start_time and end_time in text_timestamps
    df_transcription["text_timestamps"] = df_transcription[["text_timestamps", "start"]].apply(
        lambda x: get_global_timestamps(x["text_timestamps"], x["start"]), axis=1
    )

    return df_transcription


def concatenate_transcriptions(df_transcription):
    # Group by audio_path and concatenate text of the same audio file
    df_audio = (
        df_transcription.groupby("audio_path").agg(
            {
                "inference_normalized": " ".join,
                "text_timestamps": "sum",
                "date_start": "min",
                "date_end": "max",
            }
        )
    ).reset_index()

    # Add word_index to text_timestamps
    df_audio["text_timestamps"] = df_audio["text_timestamps"].apply(
        lambda x: [{"word_index": i, **timestamp} for i, timestamp in enumerate(x)]
    )

    return df_audio


def date_based_join(df, df_audio):
    """
    Join the ground truth speeches with the audio file debates based on date.
    Enumerate all possible combinations of speeches and debates where the date
    of the speech is within the date range of the debate audio file.
    """

    conn = sqlite3.connect(":memory:")
    df[["speaker_id", "speech_id", "date", "name", "party"]].to_sql("speeches", conn, index=False)
    df_audio[["audio_path", "date_start", "date_end"]].to_sql("debates", conn, index=False)

    # Make a date based join where date is within date_start and date_end
    query = """
    SELECT * FROM speeches
    JOIN debates
    ON speeches.date BETWEEN debates.date_start AND debates.date_end
    """

    df_combinations = pd.read_sql_query(query, conn)
    conn.close()

    return df_combinations


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
        return -100, -100


def preprocess(json_files):
    """
    Convenience function to process batches of json files instead of all at once.
    (memory issues with processing all at once)
    """
    transcriptions = []
    for json_file in json_files:
        transcriptions.append(read_json(json_file))
    df_transcription = preprocess_transcriptions(transcriptions)
    df_audio = concatenate_transcriptions(df_transcription)
    return df_audio


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_workers", type=int, default=12)
    argparser.add_argument("--start_year", type=int, default=1966, help="Inclusive")
    argparser.add_argument("--end_year", type=int, default=1975, help="Inclusive")
    args = argparser.parse_args()

    df = pd.read_parquet("data/riksdagen_speeches.parquet")
    # date as datetime
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    # Filter based on date
    df = df[
        (df["date"].dt.year >= args.start_year) & (df["date"].dt.year <= args.end_year)
    ].reset_index(drop=True)

    with mp.Pool(args.num_workers) as pool:
        df["anftext_normalized"] = list(
            tqdm(
                pool.imap(normalize_text, df["text"].tolist()),
                total=len(df),
                desc="Normalizing text",
            )
        )

    # read vad json
    json_files = glob.glob("data/vad_output/*.json")
    # Chunk json files into batches of 100
    json_files = [json_files[i : i + 100] for i in range(0, len(json_files), 100)]
    with mp.Pool(args.num_workers + 4) as pool:
        df_audio = pool.map(preprocess, tqdm(json_files, total=len(json_files)), chunksize=1)

    df_audio = pd.concat(df_audio, ignore_index=True)  # inference from audio

    # Possible combinations of speeches and audio files
    df_combinations = date_based_join(df, df_audio)

    # Remove speeches with duplicated speech_id and audio_path
    # (no need to align same speech against same audio more than once)
    df_combinations = df_combinations.drop_duplicates(
        subset=["speech_id", "audio_path"], keep="first"
    ).reset_index(drop=True)

    # Join in the text_timestamps and inference_normalized to df_combinations
    df_combinations = df_combinations.merge(
        df_audio[["audio_path", "text_timestamps", "inference_normalized"]],
        how="left",
        left_on="audio_path",
        right_on="audio_path",
    )
    del df_audio

    # Same speech can be duplicated because it can appear on multiple dates
    # Drop duplicates and reset index, because we are only interested in
    # joining in the normalized text once for each individual speech.
    df = df.drop_duplicates(subset="speech_id", keep="first").reset_index(drop=True)

    # Join in the anftext_normalized to df_combinations
    df_combinations = df_combinations.merge(
        df[["speech_id", "anftext_normalized", "protocol_id"]],
        how="left",
        left_on="speech_id",
        right_on="speech_id",
    )
    del df

    # Fuzzy string match the anftext_normalized (speech) and inference_normalized (audio file)
    with mp.Pool(args.num_workers + 7) as pool:
        args_fuzzy = zip(
            df_combinations["anftext_normalized"].tolist(),
            df_combinations["inference_normalized"].tolist(),
            repeat(55),  # threshold
            repeat(160),  # max_length
        )
        scores = pool.starmap(
            contiguous_fuzzy_match_split,
            tqdm(args_fuzzy, total=len(df_combinations)),
            chunksize=100,
        )

    df_combinations[["word_start", "word_end", "score"]] = pd.DataFrame(scores)
    # del scores  # Free up memory

    # Make word_start and word_end integers and allow for None/NaN
    df_combinations[["word_start", "word_end"]] = df_combinations[
        ["word_start", "word_end"]
    ].astype("Int64")

    df_combinations[["start_time", "end_time"]] = df_combinations[
        ["text_timestamps", "word_start", "word_end"]
    ].apply(get_timestamps, axis=1, result_type="expand")

    # Remove speeches where both word_start and word_end are NA
    df_aligned = df_combinations[
        ~(df_combinations["word_start"].isna() & df_combinations["word_end"].isna())
    ].reset_index(drop=True)
    del df_combinations

    df_aligned["start_or_end_missing"] = (
        df_aligned["word_start"].isna() | df_aligned["word_end"].isna()
    )
    df_aligned["more_than_one_candidate"] = df_aligned["speech_id"].duplicated(keep=False)

    # Remove speeches that have multiple candidates and are missing start or end
    df_aligned = df_aligned[
        (df_aligned["start_or_end_missing"] == False)
        & (df_aligned["more_than_one_candidate"] == True)
        | (df_aligned["start_or_end_missing"] == False)
        & (df_aligned["more_than_one_candidate"] == False)
        | (df_aligned["start_or_end_missing"] == True)
        & (df_aligned["more_than_one_candidate"] == False)
    ]
    df_aligned = df_aligned.drop(["start_or_end_missing", "more_than_one_candidate"], axis=1)
    # Group by speech id and select the example with the highest score with idxmax
    df_aligned = df_aligned.loc[df_aligned.groupby("speech_id")["score"].idxmax()]
    # Sort by index (their original order in transcripts)
    df_aligned = df_aligned.sort_index().reset_index(drop=True)
    df_aligned = df_aligned.drop(["inference_normalized", "text_timestamps"], axis=1)
    df_aligned.to_parquet(
        f"data/string_aligned_speeches_{args.start_year}_{args.end_year}.parquet", index=False
    )
