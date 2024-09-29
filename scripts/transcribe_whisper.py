import argparse
import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration

from rixvox.dataset import (
    AudioFileChunkerDataset,
    custom_collate_fn,
    make_transcription_chunks,
    read_json_parallel,
)

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/transcribe_whisper.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default="openai/whisper-large-v3")
argparser.add_argument(
    "--attn_implementation",
    type=str,
    default="flash_attention_2",
    choices=["flash_attention_2", "eager", "sdpa"],
    help="""Attention implementation to use. SDPA is default for torch>=2.1.1 in Hugging Face. 
    Otherwise eager is default. Use flash_attention_2 if you have installed the flash-attention package.""",
)
argparser.add_argument("--gpu_id", type=int, default=0)
argparser.add_argument(
    "--num_shards",
    type=int,
    default=1,
    help="Number of splits to make for the data. Set to the number of GPUs used.",
)
argparser.add_argument(
    "--data_shard",
    type=int,
    default=0,
    help="Which split of the data to process. 0 to num_shards-1.",
)
argparser.add_argument("--max_length", type=int, default=185)
argparser.add_argument(
    "--overwrite_all", action="store_true", help="Overwrite all existing transcriptions."
)
argparser.add_argument(
    "--overwrite_model",
    action="store_true",
    help="Overwrite existing transcriptions for the model.",
)
argparser.add_argument("--json_dir", type=str, default="data/vad_output")
argparser.add_argument("--output_dir", type=str, default="data/vad_output")
argparser.add_argument(
    "--audio_dir",
    type=str,
    default="/home/fatrek/data_network/delat/audio/riksdagen/data/riksdagen_old/all",
)
argparser.add_argument(
    "--skip_already_transcribed",
    action="store_true",
    help="""Skip already transcribed json files that exist in the output directory 
    (assumes you are using a different directory for the output).""",
    default=False,
)

args = argparser.parse_args()

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

# read vad json
logger.info("Reading json-file list")
json_files = glob.glob(f"{args.json_dir}/*.json")

if args.skip_already_transcribed:
    already_transcribed = glob.glob(f"{args.output_dir}/*.json")
    already_transcribed = [os.path.basename(f) for f in already_transcribed]
    json_files = [f for f in json_files if os.path.basename(f) not in already_transcribed]

# Split audio files to N parts if using N GPUs and select the part to process
json_files = np.array_split(json_files, args.num_shards)[args.data_shard]
json_dicts = read_json_parallel(json_files, num_workers=10)

audio_files = []
for json_dict in json_dicts:
    if "audio_file" in json_dict["metadata"]:
        audio_files.append(json_dict["metadata"]["audio_file"])
    elif "audio_path" in json_dict["metadata"]:
        path = Path(json_dict["metadata"]["audio_path"])
        # audio_files.append(os.path.join(path.parent.name, path.name))
        audio_files.append(os.path.join(path.name))


model = WhisperForConditionalGeneration.from_pretrained(
    args.model_name,
    attn_implementation=args.attn_implementation,
    torch_dtype=torch.float16,
    device_map=device,
)


def my_filter(x):
    return True


audio_dataset = AudioFileChunkerDataset(
    audio_paths=audio_files,
    json_paths=json_files,
    model_name=args.model_name,
    audio_dir=args.audio_dir,
    my_filter=my_filter,
)

# Create a torch dataloader
dataloader_datasets = torch.utils.data.DataLoader(
    audio_dataset,
    batch_size=1,
    num_workers=2,
    collate_fn=custom_collate_fn,
    shuffle=False,
)

for dataset_info in tqdm(dataloader_datasets):
    try:
        if dataset_info is None:
            # If audio file is corrupted and non-readable.
            # Audio file names logged in logs/error_audio_files.txt
            continue

        if dataset_info[0]["is_transcribed_same_model"]:
            logger.info(f"Already transcribed: {dataset_info[0]['json_path']}.")
            continue  # Skip already transcribed videos

        logger.info(f"Transcribing: {dataset_info[0]['json_path']} on device {device}.")

        dataset = dataset_info[0]["dataset"]
        dataloader_mel = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            num_workers=4,
            pin_memory=True,
            pin_memory_device=f"cuda:{args.gpu_id}",
            shuffle=False,
        )

        transcription_texts = []

        for batch in dataloader_mel:
            batch = batch.to(device).half()
            predicted_ids = model.generate(
                batch,
                return_dict_in_generate=True,
                task="transcribe",
                language="sv",
                output_scores=True,
                max_length=args.max_length,
            )
            transcription = audio_dataset.processor.batch_decode(
                predicted_ids["sequences"], skip_special_tokens=True
            )

            transcription_chunk = make_transcription_chunks(transcription, args.model_name)
            transcription_texts.extend(transcription_chunk)

        # Add transcription to the json file
        sub_dict = dataset.sub_dict
        assert len(sub_dict["chunks"]) == len(transcription_texts)

        for i, chunk in enumerate(sub_dict["chunks"]):
            if args.overwrite_all or "transcription" not in chunk:
                chunk["transcription"] = [transcription_texts[i]]
            elif "transcription" in chunk:
                if args.overwrite_model:
                    for j, transcription in enumerate(chunk["transcription"]):
                        if transcription["model"] == args.model_name:
                            chunk["transcription"][j] = transcription_texts[i]
                else:
                    models = [transcription["model"] for transcription in chunk["transcription"]]
                    # Check if transcription already exists for the model
                    if args.model_name not in models:
                        chunk["transcription"].append(transcription_texts[i])

        # Save the json file encode as utf-8
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, os.path.basename(dataset_info[0]["json_path"]))
        with open(json_path, mode="w", encoding="utf-8") as f:
            json.dump(sub_dict, f, ensure_ascii=False, indent=2, ignore_nan=True)

        logger.info(f"Transcription finished: {json_path} on {device}.")
    except Exception as e:
        logger.error(f"Transcription failed: {json_path}. Exception was {e}")
        continue
