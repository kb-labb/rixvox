import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import simplejson as json
import torch
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor

from rixvox.alignment import get_word_timestamps_hf
from rixvox.dataset import (
    AudioFileChunkerDataset,
    custom_collate_fn,
    make_transcription_chunks_w2v,
    read_json_parallel,
    wav2vec_collate_fn,
)

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/transcribe_w2v.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default="KBLab/wav2vec2-large-voxrex-swedish")
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
argparser.add_argument(
    "--overwrite_all", action="store_true", help="Overwrite all existing transcriptions."
)
argparser.add_argument(
    "--overwrite_model",
    action="store_true",
    help="Overwrite existing transcriptions for the model.",
)
argparser.add_argument(
    "--json_dir",
    type=str,
    default="data/riksdagen_web/langdetect_output_web",
)
argparser.add_argument(
    "--output_dir",
    type=str,
    default="data/riksdagen_web/vad_wav2vec_output_web",
)
argparser.add_argument(
    "--audio_dir",
    type=str,
    default=None,
)
argparser.add_argument(
    "--skip_already_transcribed",
    action="store_true",
    help="""Skip already transcribed json files that exist in the output directory 
    (assumes you are using a different directory for the output).""",
    default=False,
)
argparser.add_argument(
    "--skip_word_timestamps",
    default=False,
    action="store_true",
    help="Skip word timestamps in the transcription.",
)
args = argparser.parse_args()

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

# read vad json
logger.info(f"Reading json files from {args.json_dir}.")
json_files = glob.glob(f"{args.json_dir}/*.json")

if args.skip_already_transcribed:
    already_transcribed = glob.glob(f"{args.output_dir}/*.json")
    already_transcribed = [os.path.basename(f) for f in already_transcribed]
    json_files = [f for f in json_files if os.path.basename(f) not in already_transcribed]

json_dicts = read_json_parallel(json_files, num_workers=10)
audio_files = []
empty_json_files = []

for json_dict in json_dicts:
    if len(json_dict["chunks"]) == 0:
        # Skip empty or only static audio files
        empty_json_files.append(json_dict["json_path"])
        continue

    if "audio_file" in json_dict["metadata"]:
        audio_files.append(json_dict["metadata"]["audio_file"])
    elif "audio_path" in json_dict["metadata"]:
        path = Path(json_dict["metadata"]["audio_path"])
        # audio_files.append(os.path.join(path.parent.name, path.name))
        audio_files.append(os.path.join(path.name))


json_files = [json_file for json_file in json_files if json_file not in empty_json_files]
json_files = np.array_split(json_files, args.num_shards)[args.data_shard]

audio_files = [audio_file for audio_file in audio_files if audio_file not in empty_json_files]
audio_files = np.array_split(audio_files, args.num_shards)[args.data_shard]


model = AutoModelForCTC.from_pretrained(args.model_name, torch_dtype=torch.float16).to(device)
model.eval()


def my_filter(x):
    return True


audio_dataset = AudioFileChunkerDataset(
    audio_paths=audio_files,
    json_paths=json_files,
    model_name=args.model_name,
    audio_dir=args.audio_dir,
    my_filter=my_filter,
)

processor = Wav2Vec2Processor.from_pretrained(
    args.model_name, sample_rate=16000, return_tensors="pt"
)

dataloader_datasets = torch.utils.data.DataLoader(
    audio_dataset,
    batch_size=1,
    collate_fn=custom_collate_fn,
    num_workers=3,
    shuffle=False,
)

TIME_OFFSET = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate

for dataset_info in tqdm(dataloader_datasets):
    try:
        if dataset_info is None:
            continue

        logging.info(f"Transcribing: {dataset_info[0]['json_path']} on {device}.")

        if dataset_info[0]["is_transcribed_same_model"]:
            logger.info(f"Already transcribed: {dataset_info[0]['json_path']}.")
            continue  # Skip already transcribed videos

        dataset = dataset_info[0]["dataset"]
        dataloader_mel = torch.utils.data.DataLoader(
            dataset,
            batch_size=16,
            num_workers=4,
            collate_fn=wav2vec_collate_fn,
            pin_memory=True,
            pin_memory_device=f"cuda:{args.gpu_id}",
            shuffle=False,
        )

        transcription_texts = []

        for batch in dataloader_mel:
            batch = batch.to(device).half()
            with torch.inference_mode():
                logits = model(batch).logits

            probs = torch.nn.functional.softmax(logits, dim=-1)  # Need for CTC segmentation
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = audio_dataset.processor.batch_decode(
                predicted_ids, output_word_offsets=True
            )

            word_timestamps = get_word_timestamps_hf(
                transcription["word_offsets"], time_offset=TIME_OFFSET
            )

            transcription_chunk = make_transcription_chunks_w2v(
                transcription["text"],
                word_timestamps=word_timestamps,
                model_name=args.model_name,
                include_word_timestamps=not args.skip_word_timestamps,
            )
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
        json_path = os.path.join(args.output_dir, os.path.basename(dataset_info[0]["json_path"]))
        logger.error(f"Transcription failed: {json_path}. Exception was {e}")
        continue
