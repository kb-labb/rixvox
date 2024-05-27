import argparse
import glob
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from pyannote.audio import Pipeline
from tqdm import tqdm

from rixvox.dataset import VADAudioDataset

"""
Perform speaker diarization on audio files and output the results as JSON files.

If using multiple GPUs, set the number of GPUs with the `--num_gpu` argument
and set the `--gpu_id` argument to the GPU ID to use. This script will automatically
split the audio files to process based on the number of GPUs.

Example usage:
python scripts/diarization_pyannote.py --gpu_id 0 --num_gpu 8
python scripts/diarization_pyannote.py --gpu_id 1 --num_gpu 8
...
"""

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/diarization_pyannote.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
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
    "--hf_auth_token",
    type=str,
    default=None,
    help="Hugging Face authentication token for downloading models that require authentication (pyannote).",
)
args = argparser.parse_args()

device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=args.hf_auth_token
)
pipeline.to(device)

audio_files = glob.glob("/shared/delat/audio/riksdagen/data/riksdagen_old/**/**/*.mp3")
audio_files.extend(glob.glob("/shared/delat/audio/riksdagen/data/riksdagen_old/**/*.mp3"))

# Split audio files to 8 parts if using 8 GPUs and select the part to process
# based on the gpu_id argument
audio_files = np.array_split(audio_files, args.num_shards)[args.data_shard]
dataset = VADAudioDataset(audio_files)


# Custom data collator
def collate_fn(batch):
    audio = [x for x in batch if x is not None]
    return audio


dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=2
)

all_segments = []
for batch in tqdm(dataloader):
    for audio in batch:
        diarization_segments = pipeline(
            {
                "waveform": torch.from_numpy(audio).unsqueeze(0).to(torch.float32),
                "sample_rate": 16000,
            }
        )
        all_segments.append(diarization_segments)


output_dict = {}
output_dict["metadata"] = {
    "segmentation_model": pipeline.segmentation_model,
    "clustering_model": pipeline.klustering,
    "embedding_model": pipeline.embedding,
}

os.makedirs("/data/faton/riksdagen_old/data/diarization_output", exist_ok=True)
for i, segments in tqdm(enumerate(all_segments), total=len(all_segments)):
    diarization_dict = []
    output_dict["metadata"]["audio_path"] = audio_files[i]
    for segment in segments.itertracks(yield_label=True):
        diarization_dict.append(
            {
                "start": segment[0].start,
                "end": segment[0].end,
                "segment_id": segment[1],
                "speaker_id": segment[2],
                "start_hhmmss": pd.to_datetime(segment[0].start, unit="s").strftime("%H:%M:%S"),
                "end_hhmmss": pd.to_datetime(segment[0].end, unit="s").strftime("%H:%M:%S"),
            }
        )
    output_dict["chunks"] = diarization_dict
    # Extract only filename from audio path
    audio_path = audio_files[i].split("/")[-1]
    with open(
        f"/data/faton/riksdagen_old/data/diarization_output/{audio_path}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)
