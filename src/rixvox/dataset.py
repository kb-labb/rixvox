import json
import multiprocessing as mp
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Wav2Vec2Processor, WhisperProcessor

from rixvox.audio import convert_audio_to_wav


class AudioDataset(Dataset):
    """
    Takes multiple spectograms and returns one spectogram at a time.
    AudioFileChunker returns AudioDataset objects like this one so we can load
    the spectograms batch by batch with a dataloader.
    sub_dict is also included as an attribute so the main process doesn't have
    to read the json file again.
    """

    def __init__(self, spectograms, sub_dict):
        self.spectograms = spectograms
        self.sub_dict = sub_dict

    def __len__(self):
        return len(self.spectograms)

    def __getitem__(self, idx):
        return self.spectograms[idx]


class AudioFileChunkerDataset(Dataset):
    """
    Pytorch Dataset that converts audio file to wav, chunks
    audio file according to start/end times for observations
    specified in json file, and preprocesses data to spectograms.

    Args:
        audio_paths (list): List of paths to audio files
        json_paths (list): List of paths to json files
        model_name (str): Model name to use for the processor

    Returns:
        out_dict (dict): Dictionary with the following keys:
            "dataset": AudioDataset, or None if error reading audio file
            "metadata": Metadata from the json file
            "audio_path": Path to the audio file
            "json_path": Path to the json file
            "is_transcribed": Whether the audio has been transcribed
            "is_langdetected": Whether the audio has been language detected
    """

    def __init__(self, audio_paths, json_paths, model_name="openai/whisper-large-v2"):
        self.audio_paths = audio_paths
        self.json_paths = json_paths
        self.model_name = model_name
        if "whisper" in model_name:
            self.processor = WhisperProcessor.from_pretrained(model_name)
        elif "wav2vec2" in model_name:
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def __len__(self):
        return len(self.audio_paths)

    def check_if_transcribed(self, sub_dict):
        """
        We include information about whether transcription and langdetect has already been
        performed (using the same model). Useful for skipping already transcribed files.
        """
        if "transcription" in sub_dict["chunks"][0]:
            models = [t["model"] for t in sub_dict["chunks"][0]["transcription"]]
            is_transcribed = True if len(sub_dict["chunks"][0]["transcription"]) > 0 else False
            is_transcribed_same_model = self.model_name in models
            is_langdetected = any(
                [
                    ("language" in transcription)
                    for transcription in sub_dict["chunks"][0]["transcription"]
                ]
            )
        else:
            is_transcribed = False
            is_transcribed_same_model = False
            is_langdetected = False

        return is_transcribed, is_transcribed_same_model, is_langdetected

    def ms_to_frames(self, ms, sr=16000):
        return int(ms / 1000 * sr)

    def read_audio(self, audio_path):
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

    def json_chunks(self, sub_dict):
        for chunk in sub_dict["chunks"]:
            yield chunk["start"], chunk["end"]

    def audio_chunker(self, audio_path, sub_dict, sr=16000):
        audio, sr = self.read_audio(audio_path)

        for start, end in self.json_chunks(sub_dict):
            start_frame = self.ms_to_frames(start, sr)
            end_frame = self.ms_to_frames(end, sr)
            yield audio[start_frame:end_frame]

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        json_path = self.json_paths[idx]

        with open(json_path) as f:
            sub_dict = json.load(f)

        spectograms = []
        for audio_chunk in self.audio_chunker(audio_path, sub_dict):
            spectograms.append(
                self.processor(
                    audio_chunk, sampling_rate=16000, return_tensors="pt"
                ).input_features
                if "whisper" in self.model_name
                else self.processor(
                    audio_chunk, sampling_rate=16000, return_tensors="pt"
                ).input_values
            )

        if "whisper" in self.model_name:
            # Wav2vec2 processor doesn't pad up to 30s by default
            spectograms = torch.cat(spectograms, dim=0)

        mel_dataset = AudioDataset(spectograms, sub_dict)

        is_transcribed, is_transcribed_same_model, is_langdetected = self.check_if_transcribed(
            sub_dict
        )

        out_dict = {
            "dataset": mel_dataset,
            "metadata": sub_dict["metadata"],
            "audio_path": audio_path,
            "json_path": json_path,
            "is_transcribed": is_transcribed,
            "is_transcribed_same_model": is_transcribed_same_model,
            "is_langdetected": is_langdetected,
        }

        return out_dict


class VADAudioDataset(torch.utils.data.Dataset):
    def __init__(self, files, sr=16000, chunk_size=30):
        self.files = files
        self.sr = sr
        self.chunk_size = chunk_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio, sr = self.read_audio(self.files[idx])
        return audio

    def read_audio(self, audio_path):
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


def custom_collate_fn(batch: dict) -> list:
    """
    Collate function to allow dictionaries with Datasets in the batch.
    """
    # Remove None values
    batch = [b for b in batch if b is not None]

    # Return None if batch is empty
    if len(batch) == 0:
        return None

    # Return the batch
    return batch


def wav2vec_collate_fn(batch):
    """
    We need to pad the input_values to the longest sequence,
    since wav2vec2 doesn't do this by default.
    """
    # Remove None values
    batch = [b[0] for b in batch if b is not None]

    # Pad the input_values to the longest sequence
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)

    return batch


def make_transcription_chunks(
    transcriptions,
    model_name,
):
    transcription_chunks = []

    for transcript in transcriptions:
        transcription_dict = {
            "text": transcript.encode("utf-8").decode("utf-8"),
            "model": model_name,
        }
        transcription_chunks.append(transcription_dict)

    return transcription_chunks


def make_transcription_chunks_w2v(
    transcriptions,
    word_timestamps,
    model_name,
):
    transcription_chunks = []

    for i, transcript in enumerate(transcriptions):
        transcription_dict = {
            "text": transcript,
            "word_timestamps": word_timestamps[i],
            "model": model_name,
        }
        transcription_chunks.append(transcription_dict)

    return transcription_chunks


def read_json(json_path):
    with open(json_path) as f:
        sub_dict = json.load(f)
    return sub_dict


def read_json_parallel(json_paths, num_workers=6):
    with mp.Pool(num_workers) as pool:
        sub_dicts = pool.map(read_json, tqdm(json_paths, total=len(json_paths)), chunksize=20)
    return sub_dicts
