import argparse
import datetime
import glob
import logging
import os

import numpy as np
import simplejson as json
import torch
from tqdm import tqdm
from transformers import AutoProcessor, WhisperForConditionalGeneration

from rixvox.dataset import (
    AudioFileChunkerDataset,
    custom_collate_fn,
    read_json_parallel,
)
from rixvox.detect_language import get_language_probs


def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, default="openai/whisper-large-v3")
    argparser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "eager", "sdpa"],
        help="""Attention implementation to use. SDPA is default for torch>=2.1.1 in Hugging Face. 
        Otherwise eager is default. Use flash_attention_2 if you have installed the flash-attention package.""",
    )
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
    argparser.add_argument("--max_length", type=int, default=185)
    argparser.add_argument(
        "--overwrite_all", action="store_true", help="Overwrite all existing transcriptions."
    )
    argparser.add_argument(
        "--overwrite_model",
        action="store_true",
        help="Overwrite existing transcriptions for the model.",
    )
    argparser.add_argument(
        "--json_dir", type=str, default="data/speeches_by_audiofile_chunked_web"
    )
    argparser.add_argument("--batch_size", type=int, default=16)
    argparser.add_argument(
        "--audio_dir",
        type=str,
        default="data/audio/2000_2024",
    )
    argparser.add_argument("--output_dir", type=str, default="data/langdetect_output_web")
    return argparser.parse_args()


def top_n_lang(language_probs, n=5):
    lp = sorted(language_probs.items(), key=lambda x: -x[1])
    result = {l: p for l, p in lp[:n]}
    if "sv" not in result.keys():
        result["sv"] = language_probs["sv"]
    return result


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    now = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    logging.basicConfig(
        filename=f"logs/lang_detect_whisper-{now}.log",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger(__name__)

    args = get_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # read vad json
    logger.info("Reading json-file list")
    json_files = sorted(glob.glob(f"{args.json_dir}/*.json"))
    # Split audio files to N parts if using N GPUs and select the part to process
    json_files = np.array_split(json_files, args.num_shards)[args.data_shard]
    json_dicts = read_json_parallel(json_files, num_workers=10)

    audio_files = []
    for json_dict in json_dicts:
        audio_files.append(json_dict["metadata"]["audio_file"])

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # my_filter = lambda x: x["duration"] > 20_000
    def my_filter(x):
        if "language_probs" in x:
            return False
        return True

    audio_dataset = AudioFileChunkerDataset(
        audio_paths=audio_files,
        json_paths=json_files,
        model_name=args.model_name,
        audio_dir=args.audio_dir,
        my_filter=my_filter,
    )

    # Create a torch dataloader
    dataloader_datasets = torch.utils.data.DataLoader(
        audio_dataset,
        batch_size=1,
        num_workers=2,
        prefetch_factor=3,
        collate_fn=custom_collate_fn,
        shuffle=False,
    )

    logger.info("Iterate over outer dataloader")
    for dataset_info in tqdm(dataloader_datasets):
        # print(dataset_info[0]["json_path"])
        try:
            if dataset_info[0]["dataset"] is None:
                logger.info(f"Do nothing for {dataset_info[0]['json_path']}")
                continue

            dataset = dataset_info[0]["dataset"]
            dataloader_mel = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=4,
                pin_memory=True,
                pin_memory_device=f"cuda:{args.gpu_id}",
                shuffle=False,
            )

            detected_langs = []
            for batch in dataloader_mel:
                batch = batch.to(device).half()
                predicted_ids = model.generate(
                    batch,
                    return_dict_in_generate=True,
                    task="transcribe",
                    output_scores=True,
                    max_length=1,
                )
                _, _, dl = get_language_probs(predicted_ids["scores"][0])
                detected_langs.extend(dl)
                # detected_langs.extend(detect_language(model, processor.tokenizer, batch))

            # Add transcription to the json file
            sub_dict = dataset.sub_dict
            assert len(list(filter(lambda x: my_filter(x), sub_dict["chunks"]))) == len(
                detected_langs
            )

            for i, chunk in enumerate(filter(lambda x: my_filter(x), sub_dict["chunks"])):
                if args.overwrite_all or "language_probs" not in chunk:
                    chunk["language_probs"] = {args.model_name: top_n_lang(detected_langs[i])}
                elif args.model_name not in chunk["language_probs"] and not args.overwrite_model:
                    chunk["language_probs"][args.model_name] = top_n_lang(detected_langs[i])

            # Save the json file encode as utf-8
            os.makedirs(args.output_dir, exist_ok=True)
            json_path = os.path.join(
                args.output_dir, os.path.basename(dataset_info[0]["json_path"])
            )
            with open(json_path, mode="w", encoding="utf-8") as f:
                json.dump(sub_dict, f, ensure_ascii=False, indent=2, ignore_nan=True)

            logger.info(f"Transcription finished: {json_path}.")
        except Exception as e:
            logger.info(f"Transcription failed: {json_path}. Exception was {e}")
