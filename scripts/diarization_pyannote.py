import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import simplejson as json
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


def parse_args():
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
    argparser.add_argument(
        "--num_threads",
        type=int,
        default=2,
        help="Number of threads Pytorch is allowed to use. Otherwise every process may try use all threads available.",
    )
    argparser.add_argument("--audio_dir", type=str, default="data/audio/2000_2024")
    argparser.add_argument("--output_dir", type=str, default="data/diarization_output_web")
    argparser.add_argument(
        "--skip_already_diarized",
        action="store_true",
        help="Skip already diarized files that exist in the output directory",
        default=False,
    )

    args = argparser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/diarization_pyannote.log",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(args.num_threads * 2)

    logging.info(f"Device: {device}. Getting number of threads: {torch.get_num_threads()}")
    logging.info(
        f"Device: {device}. Getting number of interop threads: {torch.get_num_interop_threads()}"
    )
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=args.hf_auth_token
    )
    pipeline.to(device)

    logging.info("Loading audio files...")
    audio_files = glob.glob(f"{args.audio_dir}/**/*.mp3")
    logging.info(f"Device: {device}. Number of audio files: {len(audio_files)}")

    if args.skip_already_diarized:
        already_diarized = glob.glob(f"{args.output_dir}/*.json")
        already_diarized = [os.path.basename(f).replace("json", "mp3") for f in already_diarized]
        audio_files = [f for f in audio_files if os.path.basename(f) not in already_diarized]
    # Split audio files to 8 parts if using 8 GPUs and select the part to process
    # based on the gpu_id argument
    audio_files = np.array_split(audio_files, args.num_shards)[args.data_shard]
    metadata_dicts = []
    for audio_file in audio_files:
        metadata_dict = {
            "audio_path": audio_file,
            "audio_file": os.path.join(*Path(audio_file).parts[-2:]),
            "metadata": {},
        }
        metadata_dicts.append(metadata_dict)

    dataset = VADAudioDataset(
        metadata=metadata_dicts, audio_dir=None
    )  # Audio dir already included in audio_files

    # Custom data collator
    def collate_fn(batch):
        audio = [x for x in batch if x is not None]
        return audio

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=collate_fn, shuffle=False, num_workers=2
    )

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Device: {device}. Entering diarization loop...")
    for batch in tqdm(dataloader):
        for meta in batch:
            logging.info(f"Device: {device}. Processing audio: {meta['audio_path']}")
            audio = meta["audio"]

            if audio is None:
                diarization_segments = []
            else:
                diarization_segments = pipeline(
                    {
                        "waveform": torch.from_numpy(audio)
                        .unsqueeze(0)
                        .to(torch.float32)
                        .to(device),
                        "sample_rate": 16000,
                    }
                )

            output_dict = {}
            output_dict["metadata"] = {
                "segmentation_model": pipeline.segmentation_model,
                "clustering_model": pipeline.klustering,
                "embedding_model": pipeline.embedding,
            }

            diarization_dict = []
            output_dict["metadata"]["audio_path"] = meta["audio_file"]
            for segment in diarization_segments.itertracks(yield_label=True):
                diarization_dict.append(
                    {
                        "start": segment[0].start,
                        "end": segment[0].end,
                        "segment_id": segment[1],
                        "speaker_id": segment[2],
                        "start_hhmmss": pd.to_datetime(segment[0].start, unit="s").strftime(
                            "%H:%M:%S"
                        ),
                        "end_hhmmss": pd.to_datetime(segment[0].end, unit="s").strftime(
                            "%H:%M:%S"
                        ),
                    }
                )
            output_dict["chunks"] = diarization_dict
            # Extract only filename from audio path
            json_path = os.path.basename(meta["audio_path"]).replace(".mp3", ".json")
            with open(os.path.join(args.output_dir, json_path), "w", encoding="utf-8") as f:
                json.dump(output_dict, f, ensure_ascii=False, indent=4)
