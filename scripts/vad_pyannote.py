import glob
import json
import os
import re

import pandas as pd
import torch

# from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
from tqdm import tqdm

from rixvox.dataset import VADAudioDataset
from rixvox.vad import VoiceActivitySegmentation, merge_chunks

# Set torch device cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vad_model(
    model_name_or_path: str = "pyannote/segmentation-3.0",
    device: torch.device = torch.device("cuda"),
    min_duration_on: float = 0.1,
    min_duration_off: float = 0.1,
):
    vad_model = Model.from_pretrained(model_name_or_path).to(device)
    hyperparameters = {
        "min_duration_on": min_duration_on,
        "min_duration_off": min_duration_off,
    }
    vad_pipeline = VoiceActivitySegmentation(segmentation=vad_model, device=torch.device(device))
    vad_pipeline.instantiate(hyperparameters)
    return vad_pipeline


# Custom data collator
def collate_fn(batch):
    audio = [x for x in batch if x is not None]

    if len(audio) == 0:
        return None

    return audio


if __name__ == "__main__":

    vad_pipeline = load_vad_model(
        model_name_or_path="pyannote/segmentation-3.0",
        device=device,
        min_duration_on=0.1,
        min_duration_off=0.1,
    )

    # Create VAD dataset
    audio_files = glob.glob("data/audio/**/**/*.mp3")
    dataset = VADAudioDataset(audio_files)

    # Create a torch dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, collate_fn=collate_fn, shuffle=False, num_workers=3
    )

    # Run VAD on all audio files
    all_segments = []
    for batch in tqdm(dataloader):
        if batch is None:
            continue

        for audio in batch:
            vad_segments = vad_pipeline(
                {
                    "waveform": torch.from_numpy(audio).unsqueeze(0).to(torch.float32),
                    "sample_rate": 16000,
                }
            )
            vad_segments = merge_chunks(vad_segments, chunk_size=30)
            all_segments.append(vad_segments)

    # Save VAD output and relevant metadata to JSON
    output_dict = {}
    output_dict["metadata"] = {
        "vad_model": "pyannote/segmentation-3.0",
        "vad_onset": vad_pipeline.onset,
        "vad_offset": vad_pipeline.offset,
        "vad_min_duration_on": vad_pipeline.min_duration_on,
        "vad_min_duration_off": vad_pipeline.min_duration_off,
    }

    for i, segments in enumerate(all_segments):
        output_dict["metadata"]["audio_path"] = audio_files[i]
        output_dict["metadata"]["dates"] = re.findall(r"\d{4}-\d{2}-\d{2}", audio_files[i])
        for segment in segments:
            segment["start_hhmmss"] = pd.to_datetime(segment["start"], unit="s").strftime(
                "%H:%M:%S"
            )
            segment["end_hhmmss"] = pd.to_datetime(segment["end"], unit="s").strftime("%H:%M:%S")
            segment["start"] = int(segment["start"] * 1000)
            segment["end"] = int(segment["end"] * 1000)
        output_dict["chunks"] = segments
        os.makedirs("data/vad_output", exist_ok=True)
        # Extract only filename from audio path
        audio_path = audio_files[i].split("/")[-1].replace(".mp3", ".json")
        with open(f"data/vad_output/{audio_path}", "w") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4)
