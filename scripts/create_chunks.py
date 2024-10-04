import argparse
import glob
import logging
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import simplejson as json
from tqdm import tqdm

from rixvox.dataset import read_json_parallel
from rixvox.make_chunks import SILENCE, make_chunks, seconds_to_ms

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/chunk_creator.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument("--json_dir", type=str, default="data/speeches_by_audiofile_aligned_web")
argparser.add_argument("--output_dir", type=str, default="data/speeches_by_audiofile_chunked_web")
args = argparser.parse_args()


def add_margin_to_alignments(alignments: list, margin: int) -> list:
    """
    Add +- margin to the start and end of each alignment if the distance
    to the previous or next alignment is greater than the margin.

    Wav2vec2 uses blank tokens both for repeated tokens and silence.
    This means we cannot know exactly where word boundaries are.
    Therefore we add a small margin to each alignment.

    Args:
        alignments (list): List of alignments.
        margin (float): Margin in seconds.
    """
    for i, alignment in enumerate(alignments):
        try:
            previous_end = alignments[i - 1]["end"]
        except IndexError:
            previous_end = alignments[i]["end"]
        try:
            next_start = alignments[i + 1]["start"]
        except IndexError:
            next_start = alignments[i]["start"]
        if alignments[i]["start"] - previous_end > margin:
            alignments[i]["start"] -= margin
        if next_start - alignments[i]["end"] > margin:
            alignments[i]["end"] += margin
    if alignments[0]["start"] - margin > 0:
        alignments[0]["start"] -= margin
    return alignments


def add_silence(
    speech_dict: dict,
) -> dict:
    end = seconds_to_ms(speech_dict["alignment"][0]["start"])
    speech_with_silence = []
    speech = speech_dict.copy()
    speech_id = speech["speech_id"]
    alignments = speech["alignment"]
    for alignment in alignments:
        if end != seconds_to_ms(alignment["start"]):
            speech_with_silence.append(
                {
                    "start": end,
                    "end": seconds_to_ms(alignment["start"]),
                    "duration": seconds_to_ms(alignment["start"]) - end,
                    "text": SILENCE,
                    "duplicate": False,
                    "live": False,
                    "is_long": False,
                    "speech_id": speech_id,
                }
            )
        end = seconds_to_ms(alignment["end"])
        duration = seconds_to_ms(alignment["end"]) - seconds_to_ms(alignment["start"])
        speech_with_silence.append(
            {
                "start": seconds_to_ms(alignment["start"]),
                "end": seconds_to_ms(alignment["end"]),
                "duration": duration,
                "text": alignment["text"],
                "duplicate": False,
                "live": False,
                "is_long": duration > 30_000,
                "speech_id": speech_id,
            }
        )
    return speech_with_silence


def add_silence_and_make_chunks(speeches: list) -> list:
    for speech in speeches:
        if len(speech["alignment"]) == 0:
            continue
        speech["alignment"] = add_margin_to_alignments(speech["alignment"], margin=0.05)
        speech["alignment"] = add_silence(speech)
        # Dict will be modified in place with a "chunks" key added to each speech
        make_chunks(speech, min_threshold=1_000, silent_chunks=True, surround_silence=True)


json_files = glob.glob(f"{args.json_dir}/*")
json_dicts = read_json_parallel(json_files, num_workers=10)


for json_dict in tqdm(json_dicts):
    speeches = json_dict["speeches"]
    _ = add_silence_and_make_chunks(speeches)  # In place modification


# Remove "chunks" fom each speech and add them (combined) as a single key
# to the top level of the json_dict
for json_dict in tqdm(json_dicts):
    chunks = []
    for speech in json_dict["speeches"]:
        if "chunks" not in speech:
            continue
        chunks.extend(speech.pop("chunks"))
    json_dict["chunks"] = chunks


# Write json_dicts back to disk with mp
def write_json(json_dict, output_dir=args.output_dir):
    os.makedirs(output_dir, exist_ok=True)
    audio_file = json_dict["metadata"]["audio_file"]
    # Filename without extension

    # If audio_file is a path, extract the filename
    audio_file = Path(audio_file).name
    audio_file = os.path.splitext(audio_file)[0]
    json_path = os.path.join(output_dir, f"{audio_file}.json")
    with open(json_path, "w") as f:
        json.dump(json_dict, f, ignore_nan=True, ensure_ascii=False, indent=2)


with mp.Pool(12) as pool:
    list(tqdm(pool.imap(write_json, json_dicts, chunksize=20), total=len(json_dicts)))
