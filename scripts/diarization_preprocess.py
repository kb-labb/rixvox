import argparse
import glob
import multiprocessing as mp
import os

import pandas as pd
from tqdm import tqdm

from rixvox.dataset import read_json

argparser = argparse.ArgumentParser()
argparser.add_argument("--num_workers", type=int, default=16)
argparser.add_argument("--output_dir", type=str, default="data/riksdagen_web")
argparser.add_argument("--json_dir", type=str, default="data/diarization_output_web")
args = argparser.parse_args()


def preprocess_diarizations(diarizations):
    # Expand nested list
    df_diarization = pd.json_normalize(diarizations, "chunks", ["metadata"])
    df_diarization["clustering_model"] = df_diarization["metadata"].apply(
        lambda x: x["clustering_model"]
    )
    df_diarization["segmentation_model"] = df_diarization["metadata"].apply(
        lambda x: x["segmentation_model"]
    )
    df_diarization["embedding_model"] = df_diarization["metadata"].apply(
        lambda x: x["embedding_model"]
    )
    df_diarization["audio_path"] = df_diarization["metadata"].apply(lambda x: x["audio_path"])
    df_diarization = df_diarization.drop(columns=["metadata"])

    return df_diarization


def preprocess(json_files):
    """
    Convenience function to process batches of json files instead of all at once.
    (memory issues with processing all at once)
    """
    diarizations = []
    for json_file in json_files:
        diarizations.append(read_json(json_file))

    df_diarization = preprocess_diarizations(diarizations)
    return df_diarization


json_files = glob.glob(os.path.join(args.json_dir, "*.json"))
json_files = [json_files[i : i + 100] for i in range(0, len(json_files), 100)]

# Process diarization files in parallel

with mp.Pool(args.num_workers) as pool:
    df_list = list(tqdm(pool.imap(preprocess, json_files), total=len(json_files)))


df_diarization = pd.concat(df_list).reset_index(drop=True)
df_diarization.to_parquet(os.path.join(args.output_dir, "df_diarization.parquet"), index=False)
