import pandas as pd

from rixvox.dataset import read_json
from rixvox.text import normalize_text

"""
Functions to preprocess word level transcriptions in json to a pandas DataFrame.
"""


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


def preprocess_json(json_files):
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
