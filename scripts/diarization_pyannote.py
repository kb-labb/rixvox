import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from pyannote.audio import Pipeline
from tqdm import tqdm

from rixvox.dataset import VADAudioDataset

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
pipeline.to(torch.device("cuda"))

audio_files = glob.glob("data/**/*.mp3")
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
    os.makedirs("data/diarization_output", exist_ok=True)
    # Extract only filename from audio path
    audio_path = audio_files[i].split("/")[-1]
    with open(f"data/diarization_output/{audio_path}.json", "w") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=4)
