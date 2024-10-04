import argparse
import glob
import logging
import multiprocessing as mp
import sqlite3
from itertools import repeat

import pandas as pd
from tqdm import tqdm

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

        if len(row["text_timestamps"]) < (row["word_end"] - 1):
            # In rare cases, the word_end is slightly larger than the length of text_timestamps.
            # No time to debug what is causing this, so just return the last timestamp
            return (
                row["text_timestamps"][row["word_start"]]["start_time"],
                row["text_timestamps"][-1]["end_time"],
            )

        return -100, -100


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--num_workers", type=int, default=12)
    argparser.add_argument("--start_year", type=int, default=1966, help="Inclusive")
    argparser.add_argument("--end_year", type=int, default=2002, help="Inclusive")
    argparser.add_argument(
        "--second_pass",
        action="store_true",
        help="Run second pass on speeches that were not aligned in first pass.",
    )
    argparser.add_argument(
        "--date_margin_addition",
        type=int,
        default=21,
        help="Number of days to add to start and end date range to expand the search space.",
    )
    args = argparser.parse_args()

    df = pd.read_parquet("data/riksdagen_speeches_new.parquet")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Filter based on date to process only a subset of the data at a time (memory issues)
    df = df[
        (df["date"].dt.year >= args.start_year) & (df["date"].dt.year <= args.end_year)
    ].reset_index(drop=True)

    if args.second_pass:
        logger.info("Running second pass on speeches that were not aligned in first pass.")
        df_aligned = pd.read_parquet(f"data/string_aligned_speeches.parquet")
        # Filter out speeches that were aligned in first pass
        df = df[~df["speech_id"].isin(df_aligned["speech_id"])].reset_index(drop=True)
        del df_aligned

    with mp.Pool(args.num_workers) as pool:
        df["text_normalized"] = list(
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
        df_audio = pool.map(preprocess_json, tqdm(json_files, total=len(json_files)), chunksize=1)

    df_audio = pd.concat(df_audio, ignore_index=True)  # inference from audio
    # Someone purposefully set invalid dates for faulty audio files (we set them to 2050-05-05)
    df_audio.loc[
        df_audio["audio_path"].str.contains("RD_EN_L_1972-29-05_1972-29-05.1.mp3"), "date_start"
    ] = "2050-05-05"
    df_audio.loc[
        df_audio["audio_path"].str.contains("RD_EN_L_1972-29-05_1972-29-05.1.mp3"), "date_end"
    ] = "2050-05-05"
    df_audio.loc[
        df_audio["audio_path"].str.contains("RD_EN_L_1972-29-05_1972-29-05.2.mp3"), "date_start"
    ] = "2050-05-05"
    df_audio.loc[
        df_audio["audio_path"].str.contains("RD_EN_L_1972-29-05_1972-29-05.2.mp3"), "date_end"
    ] = "2050-05-05"
    df_audio.loc[
        df_audio["audio_path"].str.contains(
            "RD_EN_A_1996-19-23_1996-10-24_1996-10-25_1996-10-29.mp3"
        ),
        "date_start",
    ] = "1996-10-23"
    df_audio["date_start"] = pd.to_datetime(df_audio["date_start"], format="%Y-%m-%d")
    df_audio["date_end"] = pd.to_datetime(df_audio["date_end"], format="%Y-%m-%d")

    if args.second_pass:
        # Subtract the date_margin_addition from date_start and add date_margin_addition to date_end
        df_audio["date_start"] = df_audio["date_start"] - pd.Timedelta(
            days=args.date_margin_addition
        )
        df_audio["date_end"] = df_audio["date_end"] + pd.Timedelta(days=args.date_margin_addition)

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

    # Join in the text_normalized to df_combinations
    df_combinations = df_combinations.merge(
        df[["speech_id", "text_normalized", "protocol_id"]],
        how="left",
        left_on="speech_id",
        right_on="speech_id",
    )
    del df

    # Fuzzy string match the text_normalized (speech) and inference_normalized (audio file)
    with mp.Pool(args.num_workers + 7) as pool:
        args_fuzzy = zip(
            df_combinations["text_normalized"].tolist(),
            df_combinations["inference_normalized"].tolist(),
            repeat(55),  # threshold, see contiguous_fuzzy_match_split for details
            repeat(120),  # max_length
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

    # Remove speeches that have multiple candidates and also are missing start or end
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
    df_aligned["duration"] = df_aligned["end_time"] - df_aligned["start_time"]

    if args.second_pass:
        df_aligned.to_parquet(
            f"data/string_aligned_speeches_{args.start_year}_{args.end_year}_second_pass.parquet",
            index=False,
        )
    else:
        df_aligned.to_parquet(
            f"data/string_aligned_speeches_{args.start_year}_{args.end_year}.parquet", index=False
        )
