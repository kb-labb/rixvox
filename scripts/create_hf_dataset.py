import argparse
import glob
import logging
import multiprocessing as mp
import os
import re
import unicodedata
from io import BytesIO

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, Dataset, Sequence, Value, disable_caching, load_dataset
from rapidfuzz.distance.Levenshtein import normalized_distance
from tqdm import tqdm

FINAL_COLUMN_ORDER = [
    "text",
    "audio",
    "name",
    "party",
    "gender",
    "role",
    "district",
    "year",
    "dates",
    "date_approx",
    "start",
    "end",
    "duration",
    "audio_file",
    "text_normalized",
    "text_timestamps",
    "text_previous",
    "whisper_transcription",
    "wav2vec_transcription",
    "bleu_whisper",
    "bleu_wav2vec",
    "wer_whisper",
    "wer_wav2vec",
    "cer_whisper_first",
    "cer_wav2vec_first",
    "cer_whisper_last",
    "cer_wav2vec_last",
    "is_silence",
    "lang_prob_sv",
    "shard",
    "speaker_id",
    "riksdagen_id",
    "chunk_id",
    "speech_id",
    "protocol_id",
]


def convert_dataset_to_audio(dataset, audio_column="audio", sampling_rate=16000, num_workers=4):
    """
    Convert a float sequence column in a dataset to an Audio feature column.

    Args:
        dataset (Dataset): Hugging Face dataset
        audio_column (str): Name of the column containing audio data
        sampling_rate (int): Sampling rate for the audio

    Returns:
        Dataset: New dataset with converted audio column
    """

    def convert_audio_format(example):
        # Convert the list of floats to a numpy array
        audio_array = np.array(example[audio_column], dtype=np.float32)

        # Create a BytesIO buffer to store the WAV data
        buffer = BytesIO()

        # Write the array to the buffer in WAV format
        sf.write(buffer, audio_array, sampling_rate, format="WAV")

        # Get the bytes from the buffer
        audio_bytes = buffer.getvalue()

        return {audio_column: {"bytes": audio_bytes, "path": None}}

    # First convert the raw arrays to audio bytes
    dataset_with_bytes = dataset.map(
        convert_audio_format,
        desc="Converting to audio bytes",
        num_proc=num_workers,  # Adjust based on your CPU cores
    )

    # Now cast the column to Audio feature
    converted_dataset = dataset_with_bytes.cast_column(
        audio_column, Audio(sampling_rate=sampling_rate)
    )

    return converted_dataset


def enforce_float32(dataset, audio_column="audio", num_workers=4):
    """
    Ensure the audio array in the dataset is of type float32.

    Args:
        dataset (Dataset): Hugging Face dataset with an Audio column.
        audio_column (str): Name of the audio column.

    Returns:
        Dataset: Updated dataset with float32 arrays.
    """

    def convert_to_float32(example):
        # Convert the array to float32
        example[audio_column]["array"] = example[audio_column]["array"].astype(np.float32)
        return example

    # Apply the conversion to the dataset
    dataset = dataset.map(
        convert_to_float32,
        desc="Converting audio arrays to float32",
        num_proc=num_workers,  # Adjust based on your CPU cores
    )
    return dataset


def cer_head(text: str, transcription: str, len_lookback: int = 10) -> float:
    """
    Calculate the CER between the head of the text and the transcription based on
    number of characters (len_lookback) to look forward.
    """

    head_text = text[:len_lookback]
    head_transcription = transcription[:len_lookback]

    cer_match_head = normalized_distance(head_text, head_transcription)
    return cer_match_head


def cer_tail(text: str, transcription: str, len_lookback: int = 10) -> float:
    """
    Calculate the CER between the tail of the text and the transcription based on
    number of characters (len_lookback) to look back.
    """

    tail_text = text[-len_lookback:]
    tail_transcription = transcription[-len_lookback:]

    cer_match_tail = normalized_distance(tail_text, tail_transcription)
    return cer_match_tail


def calculate_metrics(row, score_function: callable, normalize_text: callable):
    """
    Calculate the score (bleu, wer, cer) between the normalized text and the transcriptions.

    Args:
        row: row of the DataFrame
        score_function: Function that calculates the score between two texts
        normalize_text: Function that normalizes the text
    Returns:
        score_whisper, score_wav2vec: score between the normalized text and the whisper transcription
            and score between the normalized text and the wav2vec transcription
    """
    text_normalized = row["text_normalized"]
    whisper_normalized = normalize_text(row["whisper_transcription"])
    wav2vec_normalized = normalize_text(row["wav2vec_transcription"])

    score_whisper = score_function(text_normalized, whisper_normalized)
    score_wav2vec = score_function(text_normalized, wav2vec_normalized)
    return score_whisper, score_wav2vec


def clean_text(text):
    """
    Clean the ground truth text of unwanted characters and patterns.

    Args:
        text (str): The text to normalize.
    Returns:
        str: The cleaned text.
    """

    # Replace abbreviations with their full form
    # text = expand_abbreviations(text)

    # Normalize unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove hyphens and double hyphens
    text = text.replace("- -", " ").replace("– –", " ")

    # Remove "/  /"  and everything between them
    text = re.sub(r"/.*?/", " ", text)

    # Remove everything between parentheses
    text = re.sub(r"\(.*?\)", "", text)

    # Remove everything between brackets
    text = re.sub(r"\[.*?\]", "", text)

    # Remove if HTML tag containing letters or whitespace, e.g. < p>
    # But don't target <> that has numbers or punctuation like <|2.24|>.
    text = re.sub(r"<[/a-zA-Z\s]+>", "", text)

    # Remove everything between asterisks *
    text = re.sub(r"\*.*?\*", "", text)

    # Remove "-" and "–" in the beginning of the text
    text = text.lstrip("-–").rstrip("-–")

    # Remove - and – from a string if the preceding character is >
    # (when training with timestamps like <|0.00|> in the text)
    text = re.sub(r"(?<=[>])[-–]", "", text)

    # Remove - – from a string if the following character ia <
    text = re.sub(r"[-–](?=[<])", "", text)

    # Remove ... from a string if the preceding character is >
    text = re.sub(r"(?<=[>])\.\.\.", "", text)

    # Remove hyphens and dashes in the beginning and in the end of a word, but leave hyphens in the middle of words
    text = re.sub(r"(?<=\s)[-–](?=\w)|(?<=\w)[-–](?=\s)", "", text)

    # Remove hyphens in examples such as -...med...
    text = re.sub(r"(?<=\s)[-–](?=\.\.\.)", "", text)

    # Use regex to remove '...' from the beginning of any text
    text = re.sub(r"^\.\.\.", "", text)

    # Remove ... ... and replace with single space
    text = re.sub(r"(.+)\.\.\. \.\.\.", r"\1 ", text)

    # Remove multiple spaces, newlines, and breaks, and replace with single space
    text = re.sub(r"\s+", " ", text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text


def add_prev_text_column(df):
    """
    Add previous text column if the timestamps from consecutive rows match.
    Previous text can be used as prompt during training.
    """

    df = df.sort_values(["speech_id", "chunk_id"])
    df["end_prev"] = df["end"].shift(1)

    # If "start" of the current row is equal to "end_prev" then insert the previous row's text
    df["prev_text_bool"] = df["start"] == df["end_prev"]
    df["text_previous"] = df["text"].shift(1)
    # If previous text is <|nospeech|> or "", then prev_text_bool should be False
    df.loc[df["text_previous"] == "<|nospeech|>", "prev_text_bool"] = False
    df.loc[df["text_previous"] == "", "prev_text_bool"] = False
    df.loc[~df["prev_text_bool"], "text_previous"] = None
    df = df.reset_index(drop=True)

    return df


def preprocess_dataset(filepath, new_filepath):
    """
    Args:
        filepath: Filepath to the parquet shard
        new_filepath: Filepath to save the new parquet shard
    """
    df = pd.read_parquet(filepath)

    # Clean the text to be more consistently formatted
    df["text"] = df["text"].apply(clean_text)
    df["n_words"] = df["text"].apply(lambda x: len(x.split()))
    df["text_timestamps"] = df["text_whisper"].apply(clean_text)

    # Our timestamp formatting is incorrect, we need <|x.xx|> instead of <x.xx>
    df["text_timestamps"] = df["text_timestamps"].apply(
        lambda x: re.sub(r"(?<=[<])(\d{1,2}\.\d{2})(?=[>])", r"|\1|", x)
    )

    # non-speech segment boolean
    df["is_silence"] = df.apply(
        lambda x: x["text"].strip() == "" and x["wav2vec_transcription"] == "", axis=1
    )

    # Add previous text column to use as prompt during training
    df = add_prev_text_column(df)
    df.to_parquet(new_filepath, row_group_size=100)


def remap_filepaths(filepaths):
    """
    Our dataset is split into two sources: riksdagen_old and riksdagen_web.
    The shards in both sources are numbered separately from 0000 and up.
    We want to unify the numbering and name them rixvox_0000.parquet and up.

    Args:
        filepaths: List of filepaths to the parquet shards from both sources.
    """
    df = pd.DataFrame(filepaths, columns=["filepaths"])
    df["old_filename"] = df["filepaths"].apply(lambda x: os.path.basename(x))
    df["data_source"] = df["old_filename"].apply(
        lambda x: "riksdagen_old" if "old" in x else "riksdagen_web"
    )
    df["shard_name"] = df["filepaths"].apply(
        lambda x: os.path.basename(x).split("_")[-1].split(".")[0]
    )
    df["shard_int"] = df["shard_name"].apply(lambda x: int(x))
    # Add max shard number of riksdagen_old to all riksdagen_web shards for unified numbering
    max_shard_number = df.loc[df["data_source"] == "riksdagen_old", "shard_int"].max()
    df.loc[df["data_source"] == "riksdagen_web", "shard_int"] += max_shard_number + 1
    df["shard_name_new"] = df["shard_int"].apply(lambda x: f"{x:04d}")
    df["new_filename"] = df.apply(lambda x: f"rixvox_{x['shard_name_new']}.parquet", axis=1)
    df["new_filepath"] = df.apply(
        lambda x: os.path.join(args.output_dir, x["new_filename"]), axis=1
    )

    return df


if __name__ == "__main__":
    logging.basicConfig(
        filename="logs/create_hf_dataset.log",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, default="/data/faton")
    argparser.add_argument("--output_dir", type=str, default="/data/faton/parquet")
    argparser.add_argument("--cache_dir", type=str, default=None)
    argparser.add_argument("--num_workers", type=int, default=12)
    args = argparser.parse_args()

    disable_caching()
    os.makedirs(args.output_dir, exist_ok=True)

    # list parquet shards
    riksdagen_old = glob.glob(
        f"{args.data_dir}/riksdagen_old/data/parquet/riksdagen_old_*.parquet"
    )
    riksdagen_web = glob.glob(
        f"{args.data_dir}/riksdagen_web/data/parquet/riksdagen_web_*.parquet"
    )

    # create df with all parquet files
    filepaths = riksdagen_old + riksdagen_web
    df_filepaths = remap_filepaths(filepaths)

    def create_hf_dataset(filepath_dict, cache_dir=args.cache_dir):
        """
        Preprocess the dataset and save it to parquet.
        Load the dataset as HF dataset and convert the audio column to Audio feature.
        Calculate CER between the normalized text and the transcriptions.
        Finally, save the dataset to parquet for upload to Hugging Face Hub.

        Args:
            filepath_dict: Dictionary containing the input and output filepaths
        """

        input_filepath = filepath_dict["filepaths"]
        output_filepath = filepath_dict["new_filepath"]
        shard = filepath_dict["shard_name_new"]

        logging.info(f"Processing shard {shard} from {input_filepath}")
        preprocess_dataset(input_filepath, output_filepath)

        logging.info(f"Loading {output_filepath} as HF dataset.")
        # Load sa HF dataset
        dataset = load_dataset(
            "parquet",
            data_files=output_filepath,
            cache_dir=cache_dir,
            keep_in_memory=True,
        )

        logging.info(f"Converting audio column to Audio feature for {output_filepath}")
        dataset = convert_dataset_to_audio(
            dataset, audio_column="audio", sampling_rate=16000, num_workers=2
        )
        # dataset = enforce_float32(dataset, audio_column="audio", num_workers=2)

        # Add cer_whisper_first and cer_wav2vec_first, cer_whisper_last and cer_wav2vec_last
        dataset = dataset.map(
            lambda x: {
                "cer_whisper_first": cer_head(x["text_normalized"], x["whisper_transcription"]),
                "cer_wav2vec_first": cer_head(x["text_normalized"], x["wav2vec_transcription"]),
                "cer_whisper_last": cer_tail(x["text_normalized"], x["whisper_transcription"]),
                "cer_wav2vec_last": cer_tail(x["text_normalized"], x["wav2vec_transcription"]),
                "shard": shard,
            },
            num_proc=4,
        )

        # Change column ordering
        dataset = dataset.select_columns(FINAL_COLUMN_ORDER)

        # Write to parquet
        dataset["train"].to_parquet(output_filepath, batch_size=100)
        logging.info(f"Saved dataset to {output_filepath}")

    with mp.Pool(args.num_workers) as pool:
        # mp.pool with tqdm
        pool.map(
            create_hf_dataset,
            tqdm(
                df_filepaths[["filepaths", "new_filepath", "shard_name_new"]].to_dict(
                    orient="records"
                ),
                total=len(df_filepaths),
            ),
            chunksize=1,
        )
