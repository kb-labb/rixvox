import argparse
import glob
import logging
import multiprocessing as mp
import os
import tempfile
from datetime import datetime

import pandas as pd
import simplejson as json
import soundfile as sf
from tqdm import tqdm

from rixvox.dataset import convert_audio_to_wav
from rixvox.metrics import (
    calculate_bleu,
    calculate_wer,
    first_word_fuzzy_score,
    last_word_fuzzy_score,
)
from rixvox.text import normalize_text

logging.basicConfig(
    filename="logs/json_to_parquet.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--json_dir",
    type=str,
    default="/home/fatrek/data_network/delat/audio/riksdagen/data/whisper_output",
)
argparser.add_argument(
    "--audio_dir",
    type=str,
    default="/home/fatrek/data_network/delat/audio/riksdagen/data/riksdagen_old/all",
)
argparser.add_argument(
    "--parquet_dir",
    type=str,
    default="test",
)
argparser.add_argument(
    "--num_workers",
    type=int,
    default=2,
)
args = argparser.parse_args()


def extract_transcription(chunk):
    transcriptions = chunk["transcription"]

    for transcription in transcriptions:
        if transcription["model"] == "openai/whisper-large-v3":
            whisper_transcription = transcription["text"]
        elif transcription["model"] == "KBLab/wav2vec2-large-voxrex-swedish":
            wav2vec_transcription = transcription["text"]

    return whisper_transcription, wav2vec_transcription


def calculate_metrics(row, metric="bleu"):
    text_normalized = row["text_normalized"]
    whisper_normalized = normalize_text(row["whisper_transcription"])
    wav2vec_normalized = normalize_text(row["wav2vec_transcription"])

    if metric == "bleu":
        bleu_whisper = calculate_bleu(text_normalized, whisper_normalized)
        bleu_wav2vec = calculate_bleu(text_normalized, wav2vec_normalized)
        return bleu_whisper, bleu_wav2vec
    elif metric == "wer":
        wer_whisper = calculate_wer(text_normalized, whisper_normalized)
        wer_wav2vec = calculate_wer(text_normalized, wav2vec_normalized)
        return wer_whisper, wer_wav2vec
    elif metric == "first_word_fuzzy":
        fuzzy_whisper = first_word_fuzzy_score(text_normalized, whisper_normalized)
        fuzzy_wav2vec = first_word_fuzzy_score(text_normalized, wav2vec_normalized)
        return fuzzy_whisper, fuzzy_wav2vec
    elif metric == "last_word_fuzzy":
        fuzzy_whisper = last_word_fuzzy_score(text_normalized, whisper_normalized)
        fuzzy_wav2vec = last_word_fuzzy_score(text_normalized, wav2vec_normalized)
        return fuzzy_whisper, fuzzy_wav2vec


def json_to_df(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for chunk in data["chunks"]:
        chunk["language_probs"] = [chunk["language_probs"]]

    df = pd.json_normalize(data, record_path=["chunks"], meta=["metadata"])
    df_speeches = pd.json_normalize(data, record_path=["speeches"])
    df["text_normalized"] = df["text"].apply(normalize_text)

    df[["lang_prob_sv", "audio_file", "source"]] = df.apply(
        lambda x: pd.Series(
            [
                x["language_probs"][0]["openai/whisper-large-v3"]["sv"],
                x["metadata"]["audio_file"],
                x["metadata"]["data_source"],
            ]
        ),
        axis=1,
    )

    df[["whisper_transcription", "wav2vec_transcription"]] = df.apply(
        lambda x: pd.Series(extract_transcription(x)), axis=1
    )

    df[[f"bleu_whisper", f"bleu_wav2vec"]] = df.apply(
        lambda x: pd.Series(calculate_metrics(x, metric="bleu")), axis=1
    )

    df[[f"wer_whisper", f"wer_wav2vec"]] = df.apply(
        lambda x: pd.Series(calculate_metrics(x, metric="wer")), axis=1
    )

    df[["whisper_first", "wav2vec_first"]] = df.apply(
        lambda x: pd.Series(calculate_metrics(x, metric="first_word_fuzzy")), axis=1
    )

    df[["whisper_last", "wav2vec_last"]] = df.apply(
        lambda x: pd.Series(calculate_metrics(x, metric="last_word_fuzzy")), axis=1
    )

    df = df.drop(columns=["subs", "metadata", "transcription"])
    df = df.merge(
        df_speeches[
            [
                "speech_id",
                "protocol_id",
                "dates",
                "name",
                "party",
                "gender",
                "role",
                "district",
                "speaker_id",
                "riksdagen_id",
            ]
        ],
        on="speech_id",
        how="left",
    )

    return df


def read_audio(audio_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            convert_audio_to_wav(audio_path, os.path.join(tmpdirname, "tmp.wav"))
            audio, sr = sf.read(os.path.join(tmpdirname, "tmp.wav"))
        except Exception as e:
            print(f"Error reading audio file {audio_path}. {e}")
            os.makedirs("logs", exist_ok=True)
            with open("logs/error_audio_files.txt", "a") as f:
                f.write(f"{audio_path}\n")
            return None
    return audio, sr


def ms_to_frames(ms, sr=16000):
    return int(ms / 1000 * sr)


def get_chunks(df):
    for i, row in df.iterrows():
        yield row["start"], row["end"], row["audio_file"], row["chunk_id"]


def audio_chunker(df, audio_dir, sr=16000):
    # Group by audio file
    df_grouped = df.groupby("audio_file")
    df_files = [df_grouped.get_group(x) for x in df_grouped.groups]

    for df_file in df_files:
        audio_path = os.path.join(audio_dir, df_file["audio_file"].iloc[0])
        audio, sr = read_audio(audio_path)
        audio = audio.astype("float32")

        for start, end, audio_file, chunk_id in get_chunks(df_file):
            start_frame = ms_to_frames(start, sr)
            end_frame = ms_to_frames(end, sr)
            yield audio[start_frame:end_frame], audio_file, chunk_id


def chunks_to_parquet(df, parquet_dir="test"):
    shard_id = df["shard"].iloc[0]
    # If file exists, skip
    if os.path.exists(f"{parquet_dir}/riksdagen_old_{shard_id}.parquet"):
        return None

    chunks_generator = audio_chunker(df, audio_dir=args.audio_dir)

    logging.info(f"Processing audio chunks from shard {shard_id}")
    chunks = []
    for chunk, audio_file, chunk_id in chunks_generator:
        chunks.append({"audio": chunk, "audio_file": audio_file, "chunk_id": chunk_id})

    df_chunk = pd.DataFrame(chunks)
    df = df.merge(df_chunk, on=["audio_file", "chunk_id"], how="left")
    logging.info(f"Writing parquet file: {parquet_dir}/riksdagen_old_{shard_id}.parquet")
    df.to_parquet(f"{parquet_dir}/riksdagen_old_{shard_id}.parquet", index=False)


if __name__ == "__main__":
    json_files = glob.glob(f"{args.json_dir}/*")
    # json_dicts = read_json_parallel(json_files, num_workers=10)

    logging.info(f"Reading {len(json_files)} json files.")
    with mp.Pool(16) as pool:
        df_list = pool.map(json_to_df, tqdm(json_files), chunksize=20)

    df = pd.concat(df_list)
    df["chunk_id"] = df.index
    # Extract year from dates
    df["year"] = df["dates"].apply(lambda x: datetime.strptime(x[0], "%Y-%m-%d").year)
    df["date_approx"] = pd.to_datetime(df["dates"].apply(lambda x: x[0]), format="%Y-%m-%d")
    df = df.reset_index(drop=True)

    # Sort by minimum date in each audio file
    df["min_date"] = df.groupby("audio_file")["date_approx"].transform("min")
    df = df.sort_values(["min_date", "chunk_id"]).reset_index(drop=True)

    # Create shards for every 25 hours of audio
    df["duration_cumsum"] = df["duration"].cumsum() / 1000 / 60 / 60
    df["shard"] = (df["duration_cumsum"] // 25).astype(int)
    df["shard"] = df["shard"].astype(str).str.pad(4, side="left", fillchar="0")

    df.to_parquet("data/riksdagen_old.parquet", index=False)

    # df = pd.read_parquet("data/riksdagen_old.parquet")

    #### Make parquet files ####
    # Group by shard and split into dataframes
    df = df[
        [
            "start",
            "end",
            "duration",
            "text",
            "text_normalized",
            "text_whisper",
            "whisper_transcription",
            "wav2vec_transcription",
            "bleu_whisper",
            "bleu_wav2vec",
            "wer_whisper",
            "wer_wav2vec",
            "whisper_first",
            "wav2vec_first",
            "whisper_last",
            "wav2vec_last",
            "speech_id",
            "protocol_id",
            "sub_ids",
            "chunk_id",
            "lang_prob_sv",
            "audio_file",
            "name",
            "party",
            "gender",
            "role",
            "district",
            "speaker_id",
            "riksdagen_id",
            "dates",
            "date_approx",
            "year",
            "shard",
        ]
    ]

    df_grouped = df.groupby("shard")
    dfs = [df_grouped.get_group(x) for x in df_grouped.groups]

    os.makedirs(args.parquet_dir, exist_ok=True)
    with mp.Pool(args.num_workers) as pool:
        pool.map(chunks_to_parquet, tqdm(dfs), chunksize=1)
