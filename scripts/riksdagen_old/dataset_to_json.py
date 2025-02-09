import datetime
import glob
import multiprocessing as mp
import os

import pandas as pd
import simplejson as json
from mutagen.mp3 import MP3
from tqdm import tqdm

from rixvox.json import preprocess_json
from rixvox.metrics import calculate_bleu
from rixvox.text import normalize_text

tqdm.pandas()

"""
Scripts that processes the RixVox dataset into JSON formats to be delivered to Riksdagen and the SWERIK project.
Also adds BLEU scores to the dataset.
"""

COLUMN_OUTPUT_ORDERING = [
    "speech_id",
    "protocol_id",
    "speech_number",
    "dates",
    "name",
    "person_id",
    "speaker_id",
    "riksdagen_id",
    "party",
    "district",
    "role",
    "gender",
    "start_segment",
    "end_segment",
    "duration_segment",
    "text",
    "text_normalized",
    "transcription_w2v",
    "start_text_time",
    "end_text_time",
    "born",
    "dead",
    "bleu_score",
    "overall_score",
    "nr_speech_segments",
    "start_segment_same",
    "audio_file",
]


def get_index_from_timestamp(timestamp, text_timestamps):
    """
    Get the index of the word from the timestamp
    """
    for i, ts in enumerate(text_timestamps):
        if ts["start_time"] <= timestamp < ts["end_time"]:
            return i

        try:
            if ts["end_time"] <= timestamp < text_timestamps[i + 1]["start_time"]:
                return i + 1
        except IndexError:
            pass

    if timestamp >= text_timestamps[-1]["end_time"]:
        return len(text_timestamps) - 1

    return None


def concat_transcription_from_timestamp(begin_timestamp, end_timestamp, text_timestamps):
    """
    We use already transcribed text from wav2vec2 to get the transcription for a specific timestamp range
    instead of transcribing the audio again. We are interested in what was said between the begin_timestamp
    and end_timestamp which delimits a speech segment.
    """
    begin_index = get_index_from_timestamp(begin_timestamp, text_timestamps)
    end_index = get_index_from_timestamp(end_timestamp, text_timestamps)

    if begin_index is None or end_index is None:
        return None

    return " ".join([ts["word"] for ts in text_timestamps[begin_index : end_index + 1]])


def concat_transcription_from_timestamp_wrapper(args):
    """
    Wrapper for multiprocessing
    """
    return concat_transcription_from_timestamp(*args)


def get_speech_transcriptions(df, df_audio):
    """
    Get the wav2vec2 speech transcription for each speech segment in the dataframe
    """

    texts = []
    for i in tqdm(range(len(df))):
        text_timestamps = df_audio.loc[
            df_audio["audio_path"].str.contains(df["audio_file"][i]), "text_timestamps"
        ].values[0]
        begin_timestamp = df["start_segment"][i]
        end_timestamp = df["end_segment"][i]

        texts.append(
            concat_transcription_from_timestamp(begin_timestamp, end_timestamp, text_timestamps)
        )

    return texts


def json_by_audiofile(
    df_group, output_dir="data/speeches_by_audiofile", audio_dir="data/audio/all"
):
    df_group = df_group.sort_values("start_segment").reset_index(drop=True)
    audio_file_path = os.path.join(audio_dir, df_group["audio_file"][0])

    # Metadata field for all file level metadata
    metadata = {
        "audio_file": df_group["audio_file"][0],
        "number_of_speeches": len(df_group),
        "duration": MP3(audio_file_path).info.length,
        # "duration": df_group["duration_segment"].sum(),
        "data_source": "riksdagen",
    }

    speeches = []
    for i, speech in df_group.iterrows():
        speech_dict = speech.to_dict()
        speech_dict.pop("audio_file")
        speeches.append(speech_dict)

    json_out = {"metadata": metadata, "speeches": speeches}
    file_path = os.path.join(output_dir, f"{metadata['audio_file'].replace('.mp3', '.json')}")

    with open(file_path, "w") as f:
        json.dump(
            json_out,
            f,
            indent=4,
            ensure_ascii=False,
            ignore_nan=True,
        )


def json_by_protocol(df_group, output_dir="data/speeches_by_protocol"):
    df_group = df_group.sort_values("speech_number").reset_index(drop=True)

    # Metadata field for all file level metadata
    metadata = {
        "protocol_id": df_group["protocol_id"][0],
        "number_of_speeches": len(df_group),
        "dates": df_group["dates"][0],
    }

    speeches = []
    for i, speech in df_group.iterrows():
        speech_dict = speech.to_dict()
        speech_dict.pop("protocol_id")
        speech_dict.pop("dates")
        speeches.append(speech_dict)

    json_out = {"metadata": metadata, "speeches": speeches}
    file_path = os.path.join(output_dir, f"{metadata['protocol_id']}.json")

    with open(file_path, "w") as f:
        json.dump(
            json_out,
            f,
            indent=4,
            ensure_ascii=False,
            ignore_nan=True,
        )


if __name__ == "__main__":
    df = pd.read_parquet("data/rixvox-alignments.parquet")

    json_files = glob.glob("data/vad_output/*.json")
    # Chunk json files into batches of 100
    json_files = [json_files[i : i + 100] for i in range(0, len(json_files), 100)]
    with mp.Pool(16) as pool:
        df_audio = pool.map(preprocess_json, tqdm(json_files, total=len(json_files)), chunksize=1)

    df_audio = pd.concat(df_audio, ignore_index=True)  # inference from audio
    texts_w2v = get_speech_transcriptions(df, df_audio)

    df["transcription_w2v"] = texts_w2v
    df["transcription_w2v"] = df["transcription_w2v"].str.lower()
    df["text_normalized"] = df["text"].progress_apply(normalize_text)

    df["bleu_score"] = df[["text_normalized", "transcription_w2v"]].progress_apply(
        lambda x: calculate_bleu(x["text_normalized"], x["transcription_w2v"]), axis=1
    )

    df.to_parquet("data/rixvox-alignments_bleu.parquet", index=False)

    #### Output to json ####
    df = pd.read_parquet("data/rixvox-alignments_bleu.parquet")
    df_riksdag = pd.read_parquet("data/riksdagen_speeches_new.parquet")
    # Aggregate all possible dates for each speech_id as a list
    df_riksdag["date"] = df_riksdag["date"].dt.strftime("%Y-%m-%d")
    df_riksdag["dates"] = df_riksdag["speech_id"].map(
        df_riksdag.groupby("speech_id")["date"].agg(list)
    )
    df_riksdag["date"] = pd.to_datetime(df_riksdag["date"])
    df_riksdag = df_riksdag[
        (df_riksdag["date"] > (pd.to_datetime(df["date"].min()) - pd.Timedelta(weeks=1)))
        & (df_riksdag["date"] < (pd.to_datetime(df["date"].max()) + pd.Timedelta(weeks=1)))
    ]
    df_riksdag = df_riksdag.groupby("speech_id").first().reset_index()

    #### JSON grouped by audio file, format suitable for Riksdagen ####
    df = df.merge(
        df_riksdag[
            ["speech_id", "role", "district", "gender", "dates", "riksdagen_id", "speech_number"]
        ],
        on="speech_id",
        how="left",
    )

    df = df.sort_values(["audio_file", "start_segment"]).reset_index(drop=True)
    df = df[COLUMN_OUTPUT_ORDERING]

    df_grouped = df.groupby("audio_file")
    df_grouped = [group for _, group in df_grouped]

    with mp.Pool(12) as pool:
        pool.map(json_by_audiofile, tqdm(df_grouped, total=len(df_grouped)), chunksize=20)

    #### JSON grouped by protocol, format suitable for SWERIK project ####
    df = pd.read_parquet("data/rixvox-alignments_bleu.parquet")
    df = df_riksdag.merge(
        df.drop(
            columns=[
                "date",
                "party",
                "protocol_id",
                "dead",
                "name",
                "text",
                "born",
                "speaker_id",
                "person_id",
            ]
        ),
        on="speech_id",
        how="left",
    )

    df = df.sort_values(["protocol_id", "speech_number"]).reset_index(drop=True)
    df = df[COLUMN_OUTPUT_ORDERING]

    df.to_parquet(os.path.join("data", "rixvox-alignments_speeches_old.parquet"), index=False)

    df_grouped = df.groupby("protocol_id")
    df_grouped = [group for _, group in df_grouped]

    with mp.Pool(12) as pool:
        pool.map(json_by_protocol, tqdm(df_grouped, total=len(df_grouped)), chunksize=20)
