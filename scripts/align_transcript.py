import argparse
import glob
import logging
import multiprocessing as mp
import os
import re

import numpy as np
import simplejson as json
import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor

from rixvox.alignment import align_with_transcript
from rixvox.dataset import read_json_parallel
from rixvox.text import normalize_text

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/transcribe_w2v.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--probs_dir", type=str, default="data/probs", help="Path to directory with probs."
)

args = argparser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCTC.from_pretrained(
    "KBLab/wav2vec2-large-voxrex-swedish", torch_dtype=torch.float16
).to(device)
processor = Wav2Vec2Processor.from_pretrained(
    "KBLab/wav2vec2-large-voxrex-swedish", sample_rate=16000, return_tensors="pt"
)

json_files = glob.glob("data/speeches_by_audiofile/*")
aligned_files = glob.glob(args.probs_dir + "/*")
aligned_files = [os.path.basename(f) + ".json" for f in aligned_files]

json_files = [f for f in json_files if os.path.basename(f) in aligned_files]
vad_dicts = read_json_parallel(json_files, num_workers=10)


def align(vad_dict):
    alignments = []
    for speech in vad_dict["speeches"]:
        transcript_tokenized = sent_tokenize(speech["text"], language="swedish")
        normalized_transcript = [normalize_text(t).upper() for t in transcript_tokenized]

        # Certain sentences/tokens are empty after normalization, remove them
        # from both the normalized transcript and the tokenized transcript.
        for i in range(len(normalized_transcript) - 1, -1, -1):
            if len(normalized_transcript[i]) == 0:
                normalized_transcript.pop(i)
                transcript_tokenized.pop(i)

        probs = np.load(os.path.join(args.probs_dir, speech["probs_file"]), allow_pickle=True)

        alignment = align_with_transcript(
            transcripts=normalized_transcript,
            probs=probs,
            audio_frames=speech["audio_frames"],
            processor=processor,
            samplerate=16000,
            chunk_size=30,
        )

        for i, segment in enumerate(alignment):
            segment["start"] += float(speech["start_segment"])
            segment["end"] += float(speech["start_segment"])
            segment["text"] = transcript_tokenized[i]

        speech["alignment"] = alignment
        alignments.append(alignment)

    json_path = os.path.join(
        "data/speeches_by_audiofile",
        os.path.splitext(vad_dict["metadata"]["audio_file"])[0] + ".json",
    )

    with open(json_path, "w") as f:
        json.dump(vad_dict, f, ensure_ascii=False, indent=4)


with mp.Pool(18) as pool:
    alignments = pool.map(align, tqdm(vad_dicts, total=len(vad_dicts)), chunksize=1)
