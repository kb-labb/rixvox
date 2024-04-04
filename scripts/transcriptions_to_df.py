import glob
import json

import janitor
import pandas as pd

df = pd.read_parquet("data/riksdagen_speeches.parquet")

# read vad json
json_files = glob.glob("data/vad_output/*.json")

audio_files = []
transcriptions = []
for json_file in json_files:
    with open(json_file) as f:
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
df_transcription["date_start"] = (
    df_transcription["dates"].apply(lambda x: x[0]).astype("datetime64[ns]")
)
df_transcription["date_end"] = (
    df_transcription["dates"].apply(lambda x: x[-1]).astype("datetime64[ns]")
)
df_transcription.rename(columns={"text_inference": "text"}, inplace=True)
df["date"] = pd.to_datetime(df["date"])


# Concat all text in the same audio file
df_test = (
    df_transcription[
        ["start", "end", "text", "date_start", "date_end", "audio_path", "text_timestamps"]
    ]
    .groupby("audio_path")
    .agg(
        {
            "text": "sum",
            "start": "min",
            "end": "max",
            "date_start": "min",
            "date_end": "max",
            "text_timestamps": lambda x: [item for sublist in x for item in sublist],
        }
    )
    .reset_index()
)


df_outer = pd.merge(
    df_test[["start", "end", "audio_path", "date_start", "date_end"]],
    df,
    how="cross",
)

df[
    (df["date"] >= min(df_transcription["date_start"]))
    & (df["date"] <= max(df_transcription["date_end"]))
]

df_outer = df_outer[
    (df_outer["date"] >= df_outer["date_start"]) & (df_outer["date"] <= df_outer["date_end"])
]
