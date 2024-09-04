import argparse
import glob
import logging
import multiprocessing as mp
import os
import re

import numpy as np
import simplejson as json
import torch
import torchaudio.functional as F
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor

from rixvox.alignment import align_with_transcript
from rixvox.dataset import read_json_parallel
from rixvox.text import get_normalized_tokens, normalize_text_with_mapping


def align_pytorch(transcripts, emissions, device):
    transcript = " ".join(transcripts)
    transcript = transcript.replace("\n", " ").upper()
    targets = processor.tokenizer(transcript, return_tensors="pt")["input_ids"]
    targets = targets.to(device)

    alignments, scores = F.forced_align(emissions, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    # scores = scores.exp()  # convert back to probability
    return alignments, scores


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
json_dicts = read_json_parallel(json_files, num_workers=10)

json_dicts[0]["speeches"][0]["text"]


mappings = []
for i, speech in enumerate(json_dicts[0]["speeches"]):
    normalized_text, mapping, original_text = normalize_text_with_mapping(speech["text"])
    normalized_mapping, normalized_tokens = get_normalized_tokens(mapping)
    mappings.append(
        {
            "original_text": original_text,
            "mapping": mapping,
            "normalized_mapping": normalized_mapping,
            "normalized_tokens": normalized_tokens,
        }
    )

alignments = []
alignment_scores = []
for mapping in mappings:
    align_probs = np.load(os.path.join(args.probs_dir, speech["probs_file"]), allow_pickle=True)
    # align_probs have dim (batches, nr_logits, vocab_size),
    # We want to stack to (1, batch_size * nr_logits, vocab_size) for the model, keeping the batch dimension
    align_probs = np.vstack(align_probs)
    align_probs = torch.tensor(align_probs, device=device).unsqueeze(0)
    tokens, scores = align_pytorch(mapping["normalized_tokens"], align_probs, device)
    alignments.append(tokens)
    alignment_scores.append(scores)


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

        try:
            alignment = align_with_transcript(
                transcripts=normalized_transcript,
                probs=probs,
                audio_frames=speech["audio_frames"],
                processor=processor,
                samplerate=16000,
                chunk_size=30,
            )
        except Exception as e:
            logger.error(f"Failed to align {speech['probs_file']}: {e}")
            logger.info(
                f"Inserting empty alignment for {speech['probs_file']} in {vad_dict['metadata']['audio_file']}"
            )
            alignment = [
                {"start_segment": None, "end_segment": None, "text": ""}
                for _ in transcript_tokenized
            ]

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
