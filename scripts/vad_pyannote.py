import argparse
import glob
import logging
import os
import re

import pandas as pd
import simplejson as json
import torch

# from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
from tqdm import tqdm

from rixvox.dataset import VADAudioDataset, custom_collate_fn
from rixvox.vad import VoiceActivitySegmentation, merge_chunks

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/vad_pyannote.log",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_name", type=str, default="pyannote/segmentation-3.0")
argparser.add_argument("--audio_dir", type=str, default="data/audio/2000_2024")
argparser.add_argument("--metadata_dir", type=str, default="data/riksdagen_web")
argparser.add_argument("--output_dir", type=str, default="data/vad_output_web")
argparser.add_argument("--dataset_source", type=str, default="riksdagen_web")

args = argparser.parse_args()

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


if __name__ == "__main__":
    logger.info(
        f"Running VAD for files in: {args.audio_dir} on dataset source: {args.dataset_source}"
    )
    logger.info(f"Loading VAD model: {args.model_name} on device: {device}")
    vad_pipeline = load_vad_model(
        model_name_or_path=args.model_name,
        device=device,
        min_duration_on=0.1,
        min_duration_off=0.1,
    )

    # Create VAD dataset
    if args.dataset_source == "riksdagen_old":
        audio_files = glob.glob(f"{args.audio_dir}/**/*.mp3")
        metadata = []
        for audio_file in audio_files:
            metadata_dict = {
                "audio_file": audio_file,
                "metadata": {"dates": re.findall(r"\d{4}-\d{2}-\d{2}", audio_file)},
            }
            metadata.append(metadata_dict)
    elif args.dataset_source == "riksdagen_web":
        df = pd.read_parquet(os.path.join(args.metadata_dir, "df_audio_metadata.parquet"))
        df = df.groupby("filename").first().reset_index()
        # filename, dok_id, and date to dict
        metadata_dicts = df[["filename", "dok_id", "anf_datum"]].to_dict(orient="records")

        metadata = []
        for metadata_dict in metadata_dicts:
            metadata_temp = {
                "audio_file": metadata_dict["filename"],
                "metadata": {
                    "dok_id": metadata_dict["dok_id"],
                    "dates": metadata_dict["anf_datum"],
                },
            }
            metadata.append(metadata_temp)

    dataset = VADAudioDataset(
        metadata,
        audio_dir=args.audio_dir,
    )

    # Create a torch dataloader
    # Keep batch size to 1 to handle corrupted audio files being filtered out
    # in collate_fn
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, collate_fn=custom_collate_fn, shuffle=False, num_workers=4
    )

    # Run VAD on all audio files
    logger.info("Running VAD on all audio files")
    os.makedirs(args.output_dir, exist_ok=True)
    for i, batch in enumerate(tqdm(dataloader)):

        for meta in batch:
            logger.info(f"Processing: {meta['audio_path']}")
            audio = meta["audio"]

            if audio is None:
                vad_segments = []
            else:
                vad_segments = vad_pipeline(
                    {
                        "waveform": torch.from_numpy(audio).unsqueeze(0).to(torch.float32),
                        "sample_rate": 16000,
                    }
                )
                vad_segments = merge_chunks(vad_segments, chunk_size=30)

            output_dict = {}
            output_dict["metadata"] = {
                "vad_model": "pyannote/segmentation-3.0",
                "vad_onset": vad_pipeline.onset,
                "vad_offset": vad_pipeline.offset,
                "vad_min_duration_on": vad_pipeline.min_duration_on,
                "vad_min_duration_off": vad_pipeline.min_duration_off,
            }

            output_dict["metadata"]["audio_path"] = meta["audio_file"]
            output_dict["metadata"]["dates"] = meta["metadata"]["dates"]
            if args.dataset_source == "riksdagen_web":
                output_dict["metadata"]["dok_id"] = meta["metadata"]["dok_id"]

            for segment in vad_segments:
                segment["start_hhmmss"] = pd.to_datetime(segment["start"], unit="s").strftime(
                    "%H:%M:%S"
                )
                segment["end_hhmmss"] = pd.to_datetime(segment["end"], unit="s").strftime(
                    "%H:%M:%S"
                )
                segment["start"] = int(segment["start"] * 1000)
                segment["end"] = int(segment["end"] * 1000)

            output_dict["chunks"] = vad_segments

            audio_path = output_dict["metadata"]["audio_path"]
            json_filename = os.path.basename(audio_path).replace(".mp3", ".json")

            with open(f"{args.output_dir}/{json_filename}", "w", encoding="utf-8") as f:
                json.dump(output_dict, f, ensure_ascii=False, indent=4)

            logger.info(f"VAD output saved to: {args.output_dir}/{json_filename}")
