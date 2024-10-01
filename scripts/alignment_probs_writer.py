import argparse
import glob
import logging
import multiprocessing as mp
import os

import numpy as np
import simplejson as json
import torch
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor

from rixvox.alignment import calculate_w2v_output_length, segment_speech_probs
from rixvox.dataset import (
    AlignmentChunkerDataset,
    alignment_collate_fn,
    custom_collate_fn,
    read_json_parallel,
)

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/align_probs.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default="KBLab/wav2vec2-large-voxrex-swedish")
argparser.add_argument("--gpu_id", type=int, default=0)
argparser.add_argument(
    "--audio_dir", type=str, default="data/audio/2000_2024", help="Path to audio files directory."
)
argparser.add_argument(
    "--json_dir",
    type=str,
    default="data/speeches_by_audiofile_web",
    help="Path to directory containing json files.",
)
argparser.add_argument(
    "--json_outdir",
    type=str,
    default="data/speeches_by_audiofile_probs",
)
argparser.add_argument(
    "--probs_outdir", type=str, default="data/probs", help="Path to output directory for probs."
)
argparser.add_argument(
    "--chunk_size", type=int, default=30, help="Number of seconds the audio was chunked by."
)
argparser.add_argument(
    "--sample_rate", type=int, default=16000, help="Sample rate of the audio files."
)
argparser.add_argument(
    "--data_shard",
    type=int,
    default=0,
    help="Which split of the data to process. 0 to num_shards-1.",
)

argparser.add_argument(
    "--num_shards",
    type=int,
    default=1,
    help="Number of splits to make for the data. Set to the number of GPUs used.",
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

json_files = glob.glob(os.path.join(args.json_dir, "*.json"))
if args.skip_already_transcribed:
    already_transcribed = glob.glob(f"{args.json_outdir}/*.json")
    already_transcribed = [os.path.basename(f) for f in already_transcribed]
    json_files = [f for f in json_files if os.path.basename(f) not in already_transcribed]

json_dicts = read_json_parallel(json_files, num_workers=12)

audio_files = []
for json_dict in json_dicts:
    audio_files.append(json_dict["metadata"]["audio_file"])
# Split audio files to 8 parts if using 8 GPUs and select the part to process
# based on the gpu_id argument
json_files = np.array_split(json_files, args.num_shards)[args.data_shard]

model = AutoModelForCTC.from_pretrained(args.model_name, torch_dtype=torch.float16).to(device)
audio_dataset = AlignmentChunkerDataset(
    audio_paths=audio_files,
    json_paths=json_files,
    model_name=args.model_name,
    audio_dir=args.audio_dir,
)

processor = Wav2Vec2Processor.from_pretrained(
    args.model_name, sample_rate=16000, return_tensors="pt"
)

dataloader_datasets = torch.utils.data.DataLoader(
    audio_dataset,
    batch_size=1,  # Always keep this as 1, only change num_workers
    collate_fn=custom_collate_fn,
    num_workers=2,
    shuffle=False,
)

for dataset_info in tqdm(dataloader_datasets):
    if dataset_info is None:
        continue

    logging.info(f"Creating output probs for: {dataset_info[0]['json_path']} on {device}.")

    dataset = dataset_info[0]["dataset"]
    dataloader_mel = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
        collate_fn=alignment_collate_fn,
        pin_memory=True,
        pin_memory_device=f"cuda:{args.gpu_id}",
        shuffle=False,
    )

    current_speech_id = dataset[0][1]
    speech_index = 0
    probs_list = []
    speech_ids = []
    sub_dict = dataset.sub_dict

    for batch in dataloader_mel:
        spectograms = batch["spectograms"].to(device).half()
        with torch.inference_mode():
            logits = model(spectograms).logits

        probs = torch.nn.functional.softmax(logits, dim=-1)  # Need for alignment
        probs = probs.cpu().numpy()

        if probs.shape[0] == 1:
            # Pad the second dimension up to the nr_logits that args.chunk_size * args.sample_rate yields.
            # Usually collate_fn takes care of this when batch contains more
            # than 1 obs, but we need to handle the case when a batch contains only 1 obs.
            nr_logits = calculate_w2v_output_length(
                args.chunk_size * args.sample_rate, args.chunk_size
            )
            probs = np.pad(
                array=probs,
                pad_width=(
                    (0, 0),
                    (0, nr_logits - probs.shape[1]),  # Add remaining logits as padding
                    (0, 0),
                ),
                mode="constant",
            )

        probs_list.append(probs)
        speech_ids.extend(batch["speech_ids"])

    # Make audio file the folder to save the probs of all speeches contained within
    probs_dir = os.path.splitext(os.path.basename(dataset_info[0]["json_path"]))[0]
    try:
        for speech_id, probs in segment_speech_probs(probs_list=probs_list, speech_ids=speech_ids):
            probs_path = os.path.join(probs_dir, f"{speech_id}.npy")
            probs_fullpath = os.path.join(args.probs_outdir, probs_path)
            os.makedirs(os.path.dirname(probs_fullpath), exist_ok=True)

            sub_dict["speeches"][speech_index]["probs_file"] = probs_path
            speech_index += 1

            np.save(probs_fullpath, probs)
    except Exception as e:
        for speech_id, probs in zip(speech_ids, probs_list):
            logger.error(f"Speech id: {speech_id}")
            logger.error(f"Probs shape: {probs.shape}")
            logger.error(f"Failed to append probs for {current_speech_id}. Error: {e}")

    json_file = os.path.basename(dataset_info[0]["json_path"])
    json_path = os.path.join(args.json_outdir, json_file)
    logger.info(f"Writing probs to {json_path}")
    with open(dataset_info[0]["json_path"], "w") as f:
        json.dump(sub_dict, f, ensure_ascii=False, indent=4)
