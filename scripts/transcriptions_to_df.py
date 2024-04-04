import glob
import json
import re

import pandas as pd
from tqdm import tqdm

from rixalign.speech_finder import contiguous_fuzzy_match
from rixalign.text import normalize_text

df = pd.read_parquet("data/riksdagen_speeches.parquet")
df["anftext_normalized"] = df["text"].apply(normalize_text)

# read vad json
json_files = glob.glob("data/vad_output/*.json")

audio_files = []
transcriptions = []
for json_file in json_files:
    with open(json_file) as f:
        print(json_file)
        transcription = json.load(f)
        audio_files.append(transcription["metadata"]["audio_path"])
        transcriptions.append(transcription)

df_transcription = pd.json_normalize(transcription, "chunks", ["metadata"])

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


def get_global_timestamps(text_timestamps, start):
    for timestamp in text_timestamps:
        timestamp["start_time"] += start / 1000
        timestamp["end_time"] += start / 1000
    return text_timestamps


# Add start to each start_time and end_time in text_timestamps
df_transcription["text_timestamps"] = df_transcription[["text_timestamps", "start"]].apply(
    lambda x: get_global_timestamps(x["text_timestamps"], x["start"]), axis=1
)

# Group by audio_path and concatenate text
df_debate = (
    df_transcription.groupby("audio_path")
    .agg(
        {
            "inference_normalized": " ".join,
            "text_timestamps": "sum",
            "date_start": "min",
            "date_end": "max",
        }
    )
    .reset_index()
)

# Add word_index to text_timestamps
df_debate["text_timestamps"] = df_debate["text_timestamps"].apply(
    lambda x: [{"word_index": i, **timestamp} for i, timestamp in enumerate(x)]
)


import sqlite3

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
df_combinations

# Join in the text_timestamps and inference_normalized to the df_combinations
df_combinations = df_combinations.merge(
    df_debate[["audio_path", "text_timestamps", "inference_normalized"]],
    how="left",
    left_on="audio_path",
    right_on="audio_path",
)
# Join in the anftext_normalized to the df_combinations
df_combinations = df_combinations.merge(
    df[["speech_id", "anftext_normalized"]], how="left", left_on="speech_id", right_on="speech_id"
)

scores = []
for i, row in tqdm(df_combinations.iterrows(), total=len(df_combinations)):
    scores.append(
        contiguous_fuzzy_match(
            anftext_normalized=row["anftext_normalized"],
            anftext_inference=row["inference_normalized"],
        )
    )


df_combinations[0:1]
df_combinations
len(scores)

df_test = df_combinations[0:2600]
df_test[["word_start", "word_end", "score"]] = pd.DataFrame(scores[0:2600])
# Make word_start and word_end integers and allow for None/NaN
df_test[["word_start", "word_end"]] = df_test[["word_start", "word_end"]].astype("Int64")


# Get the start_time and end_time from text_timestamps for word_start and word_end
def get_timestamps(row):
    # If NAType, return None, None
    if pd.isna(row["word_start"]) or pd.isna(row["word_end"]):
        return None, None
    return (
        row["text_timestamps"][row["word_start"] - 1]["start_time"],
        row["text_timestamps"][row["word_end"] - 1]["end_time"],
    )


df_test[["start_time", "end_time"]] = df_test[["text_timestamps", "word_start", "word_end"]].apply(
    get_timestamps, axis=1, result_type="expand"
)
