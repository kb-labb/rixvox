import glob
import multiprocessing as mp
import sqlite3
from itertools import repeat

import pandas as pd
from tqdm import tqdm

from rixvox.dataset import read_json_parallel
from rixvox.speech_finder import contiguous_fuzzy_match_split
from rixvox.text import normalize_text


def get_global_timestamps(text_timestamps, start):
    for timestamp in text_timestamps:
        timestamp["start_time"] += start / 1000
        timestamp["end_time"] += start / 1000
    return text_timestamps


def preprocess_transcriptions(transcriptions):
    # Expand nested list[dict] column "transcription" to columns
    df_transcription = pd.json_normalize(transcriptions, "chunks", ["metadata"])
    df_transcription["transcription"] = df_transcription["transcription"].apply(lambda x: x[1])
    df_transcription["text"] = df_transcription["transcription"].apply(lambda x: x["text"])
    df_transcription["model"] = df_transcription["transcription"].apply(lambda x: x["model"])
    df_transcription["text_timestamps"] = df_transcription["transcription"].apply(
        lambda x: x["word_timestamps"]
    )
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
    df_debate = (
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
    df_debate["text_timestamps"] = df_debate["text_timestamps"].apply(
        lambda x: [{"word_index": i, **timestamp} for i, timestamp in enumerate(x)]
    )

    return df_debate


def date_based_join(df, df_debate):
    """
    Join the ground truth speeches with the audio file debates based on date.
    Enumerate all possible combinations of speeches and debates where the date
    of the speech is within the date range of the debate audio file.
    """

    conn = sqlite3.connect(":memory:")
    df[["speaker_id", "speech_id", "date", "name", "party"]].to_sql("speeches", conn, index=False)
    df_debate[["audio_path", "date_start", "date_end"]].to_sql("debates", conn, index=False)

    # Make a date based join where date is within date_start and date_end
    query = """
    SELECT * FROM speeches
    JOIN debates
    ON speeches.date BETWEEN debates.date_start AND debates.date_end
    """

    df_combinations = pd.read_sql_query(query, conn)
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
        return row["text_timestamps"][row["word_start"] - 1]["start_time"], None

    return (
        row["text_timestamps"][row["word_start"] - 1]["start_time"],
        row["text_timestamps"][row["word_end"] - 1]["end_time"],
    )


if __name__ == "__main__":
    df = pd.read_parquet("data/riksdagen_speeches.parquet")
    df["anftext_normalized"] = df["text"].apply(normalize_text)

    # read vad json
    json_files = glob.glob("data/vad_output/*.json")
    transcriptions = read_json_parallel(json_files, num_workers=6)

    df_transcription = preprocess_transcriptions(transcriptions)
    df_debate = concatenate_transcriptions(df_transcription)
    # Possible combinations of speeches and audio files
    df_combinations = date_based_join(df, df_debate)

    # Join in the text_timestamps and inference_normalized to df_combinations
    df_combinations = df_combinations.merge(
        df_debate[["audio_path", "text_timestamps", "inference_normalized"]],
        how="left",
        left_on="audio_path",
        right_on="audio_path",
    )
    # Join in the anftext_normalized to df_combinations
    df_combinations = df_combinations.merge(
        df[["speech_id", "anftext_normalized"]],
        how="left",
        left_on="speech_id",
        right_on="speech_id",
    )

    # Fuzzy string match the anftext_normalized (speech) and inference_normalized (audio file)
    with mp.Pool(7) as pool:
        args = zip(
            df_combinations["anftext_normalized"].tolist(),
            df_combinations["inference_normalized"].tolist(),
            repeat(55),  # threshold
            repeat(130),  # max_length
        )
        scores = pool.starmap(
            contiguous_fuzzy_match_split, tqdm(args, total=len(df_combinations)), chunksize=10
        )

    df_combinations[["word_start", "word_end", "score"]] = pd.DataFrame(scores)
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
    df_aligned = df_aligned.drop(colummns=["inference_normalized", "text_timestamps"])
    df_aligned.to_parquet("data/string_aligned_speeches.parquet", index=False)
