import argparse
import glob
import logging
import os
from pathlib import Path

import numpy as np
import simplejson as json
import torch
import torch.multiprocessing as mp
from nltk.data import load
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor

from rixvox.alignment import (
    add_timestamps_to_mapping,
    get_alignments_and_scores,
    get_sentence_alignment,
    map_text_to_tokens,
)
from rixvox.dataset import read_json_parallel

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/align_transcript.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--probs_dir", type=str, default="data/probs", help="Path to directory with probs."
)
argparser.add_argument(
    "--device",
    type=str,
    default="cuda",
    choices=["cuda", "cpu"],
    help="Device to run the model on. Default: cuda.",
)
argparser.add_argument(
    "--num_workers",
    type=int,
    default=24,
    help="Number of workers to use for multiprocessing if device is set to cpu.",
)
argparser.add_argument(
    "--input_dir",
    type=str,
    default="data/speeches_by_audiofile_probs",
    help="Path to directory with json files.",
)
argparser.add_argument(
    "--output_dir",
    type=str,
    default="data/speeches_by_audiofile_aligned_web",
    help="Path to directory to save aligned json files.",
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
    "--chunk_size",
    type=int,
    default=30,
    help="Number of seconds the audio was chunked by when performing inference in previous scripts.",
)

args = argparser.parse_args()


def align_and_save(json_dict):
    logger.info(f"Aligning {json_dict['metadata']['audio_file']}")
    mappings, json_dict = map_text_to_tokens(json_dict)
    alignments, alignment_scores = get_alignments_and_scores(
        json_dict=json_dict,
        mappings=mappings,
        processor=processor,
        probs_dir=args.probs_dir,
        device=args.device,
    )
    mappings = add_timestamps_to_mapping(
        json_dict, mappings, alignments, alignment_scores, chunk_size=args.chunk_size
    )
    json_dict = get_sentence_alignment(json_dict, mappings, tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(
        args.output_dir,
        Path(json_dict["metadata"]["audio_file"]).stem + ".json",
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)

    logger.info(f"Saved aligned json file to {json_path}")


if __name__ == "__main__":
    model = AutoModelForCTC.from_pretrained(
        "KBLab/wav2vec2-large-voxrex-swedish", torch_dtype=torch.float16
    ).to(args.device)
    processor = Wav2Vec2Processor.from_pretrained(
        "KBLab/wav2vec2-large-voxrex-swedish", sample_rate=16000, return_tensors="pt"
    )

    json_files = glob.glob(args.input_dir + "/*")
    aligned_files = glob.glob(args.probs_dir + "/*")
    aligned_files = [os.path.basename(f) + ".json" for f in aligned_files]
    # aligned_files = glob.glob(args.output_dir + "/*")
    # aligned_files = [os.path.basename(f) for f in aligned_files]
    json_files = [f for f in json_files if os.path.basename(f) in aligned_files]

    # Split audio files to N parts if using N GPUs and select the part to process
    json_files = np.array_split(json_files, args.num_shards)[args.data_shard]
    json_dicts = read_json_parallel(json_files, num_workers=10)

    tokenizer = load("tokenizers/punkt/swedish.pickle")
    # Add some abbreviations to the tokenizer
    tokenizer._params.abbrev_types.update(
        set(["d.v.s", "dvs", "fr.o.m", "kungl", "m.m", "milj", "o.s.v", "t.o.m", "milj.kr"])
    )

    if args.device == "cpu":
        torch.set_num_threads(1)  # Avoid oversubscription
        with mp.Pool(16) as pool:
            pool.map(align_and_save, tqdm(json_dicts, total=len(json_dicts)), chunksize=1)
    else:
        torch.set_num_threads(1)
        for json_dict in tqdm(json_dicts, total=len(json_dicts)):
            align_and_save(json_dict)
