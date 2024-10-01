import argparse
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument("--num_workers", type=int, default=16)
argparser.add_argument("--data_dir", type=str, default="data/riksdagen_web")

args = argparser.parse_args()


def find_contiguous_segments(df_diarization):

    df_contiguous = []
    for group_variables, df_group in tqdm(
        df_diarization.groupby(["audio_file"]),
        total=len(df_diarization.groupby(["audio_file"])),
    ):
        # Subset first and last segment of a contiguous speech sequence by the same speaker
        # within a speech audio file duration.
        df_group = df_group.copy()
        df_group = df_group[
            (df_group["label"] != df_group["label"].shift())
            | (df_group["label"] != df_group["label"].shift(-1))
        ]

        # Unique speech segment ids for each anforande. E.g. if speaker_0 starts, followed by
        # speaker_1, followed by speaker_0 again, then the speech segment ids are 0, 1, 2.
        df_group["speech_segment_id"] = (
            df_group["label"].eq(0) | (df_group["label"] != df_group["label"].shift())
        ).cumsum()

        df_contiguous.append(df_group)

    df_contiguous = pd.concat(df_contiguous)

    df_contiguous["nr_speech_segments"] = df_contiguous.groupby(["audio_file"])[
        "speech_segment_id"
    ].transform("max")
    # Start of contiguous speech segment
    df_contiguous["start_segment"] = df_contiguous.groupby(["audio_file", "speech_segment_id"])[
        "start"
    ].transform("min")
    # End of speech segment
    df_contiguous["end_segment"] = df_contiguous.groupby(["audio_file", "speech_segment_id"])[
        "end"
    ].transform("max")

    return df_contiguous


def find_overlaps(df_fuzzy_group, df_segments, min_overlap=0.0000001):
    """
    For each audio file, make a pass over all the fuzzy matched speeches to find
    segments that overlap with the diarization segments. Assign a speech_id and
    other relevant metadata to the segments, and keep only the segments that
    have some non-zero overlap with the diarization segments.

    Args:
    df_fuzzy_group (pd.DataFrame): Grouped dataframe of fuzzy matched speeches
    df_segments (pd.DataFrame): Diarization segments
    min_overlap (float): Minimum overlap ratio to consider a segment as overlapping

    Returns:
    df_overlap (list): List of overlapping segments
    """
    df_overlap = []
    df_segment_group = df_segments[
        df_segments["audio_file"] == df_fuzzy_group["audio_file"].values[0]
    ]

    for i, row in df_fuzzy_group.iterrows():
        df_segment_group = df_segment_group.copy()
        df_segment_group["speech_id"] = row["speech_id"]
        df_segment_group["speaker_id"] = row["riksdagen_id"]
        df_segment_group["protocol_id"] = row["protocol_id"]
        df_segment_group["text_normalized"] = row["text_normalized"]
        df_segment_group["start_text_time"] = row["start_time"]
        df_segment_group["end_text_time"] = row["end_time"]
        df_segment_group["fuzzy_score"] = row["score"]

        df_segment_group["duration_text"] = (
            df_segment_group["end_text_time"] - df_segment_group["start_text_time"]
        )
        df_segment_group["duration_overlap"] = df_segment_group[
            ["end_segment", "end_text_time"]
        ].min(axis=1) - df_segment_group[["start_segment", "start_text_time"]].max(axis=1)

        df_segment_group["overlap_ratio"] = (
            df_segment_group["duration_overlap"] / df_segment_group["duration_text"]
        )

        overlapping_segments = df_segment_group[df_segment_group["overlap_ratio"] > min_overlap]
        df_overlap.append(overlapping_segments)

    df_overlap = pd.concat(df_overlap, ignore_index=True)
    return df_overlap


def find_overlaps_multi(args):
    """
    Wrapper function to pass multiple arguments to the find_overlaps function.
    Workaround to use tqdm and imap with more than one argument.
    """
    return find_overlaps(*args)


def normalize_above_1(value):
    """
    1 is our ideal score, but we may have values that are both above and below 1.
    Normalize/rescale the values that are above 1 to 1/x^2 so that 1 is the maximum value.
    This way we punish the segments that have high recall but low precision.
    """
    if value > 1:
        return 1 / value**2
    return value


def speaker_ratio_in_speeches(df_overlap):
    """
    Calculate the ratio of each speaker's speech duration within each speech_id,
    so we can filter out segments with low speaker duration ratio.

    Args:
    df_overlap (pd.DataFrame): DataFrame with overlapping segments. The same speech_id
    may have multiple segments with different (candidate) speaker labels.

    Returns:
    df_overlap (pd.DataFrame): DataFrame with speaker ratios
    """
    # Total diarized speech segment duration within each speech_id
    df_overlap["total_overlap"] = df_overlap.groupby(["audio_file", "speech_id"])[
        "duration_overlap"
    ].transform("sum")

    # Duration of speech for each speaker (label) within each speech_id
    df_overlap["total_speaker_duration"] = df_overlap.groupby(
        ["audio_file", "speech_id", "label"]
    )["duration_segment"].transform("sum")

    # Speaker duration ratio within each speech_id with respect to the total duration of the speech_id
    df_overlap["speaker_ratio"] = (
        df_overlap["total_speaker_duration"] / df_overlap["total_overlap"]
    )

    df_overlap["speaker_ratio_normalized"] = df_overlap["speaker_ratio"].apply(normalize_above_1)

    return df_overlap


def subset_first_last_segment(df_overlap):
    """
    After removing segments with speaker_ratio less than 0.1, we may have multiple contiguous segments
    from the same speech_id that have the same speaker label.
    We subset the first and last segment of a contiguous speech sequence by the same speaker
    within each speech_id. This is done by checking if the speaker label is different from the previous
    """

    df_list = []
    for group_variables, df_group in tqdm(df_overlap.groupby(["audio_file", "speech_id"])):
        df_group = df_group.copy()
        df_group = df_group[
            (df_group["label"] != df_group["label"].shift())
            | (df_group["label"] != df_group["label"].shift(-1))
        ]
        df_list.append(df_group)

    df_overlap = pd.concat(df_list)

    return df_overlap


if __name__ == "__main__":
    df_fuzzy = pd.read_parquet(os.path.join(args.data_dir, "string_aligned_speeches.parquet"))
    df_fuzzy = df_fuzzy[~df_fuzzy["speech_id"].duplicated(keep="first")].reset_index(drop=True)
    df_diarization = pd.read_parquet(os.path.join(args.data_dir, "df_diarization.parquet"))

    df_fuzzy["speech_id"] = df_fuzzy["dok_id"] + "-" + df_fuzzy["anf_nummer"].astype(str)
    # dokid to protocol_id
    df_fuzzy = df_fuzzy.rename(columns={"dok_id": "protocol_id"})
    df_fuzzy["speaker_id"] = df_fuzzy["riksdagen_id"]
    df_fuzzy["audio_file"] = df_fuzzy["audio_path"].apply(
        lambda x: os.path.join(*Path(x).parts[-2:])
    )
    df_diarization["audio_file"] = df_diarization["audio_path"].apply(
        lambda x: os.path.join(*Path(x).parts[-2:])
    )
    df_diarization["label"] = df_diarization["speaker_id"]  # Rename speaker_id to label
    df_diarization = df_diarization.drop(columns=["speaker_id"])

    df_diarization["duration"] = df_diarization["end"] - df_diarization["start"]
    df_diarization = df_diarization[df_diarization["duration"] >= 0.6].reset_index(drop=True)

    #### Find contiguous speech segments ####
    df_contiguous = find_contiguous_segments(df_diarization)

    # Keep only one row per contiguous speech segment
    df_segments = (
        df_contiguous.groupby(["audio_file", "label", "speech_segment_id"]).first().reset_index()
    )
    df_segments = df_segments.sort_values(["audio_file", "speech_segment_id"]).reset_index(
        drop=True
    )
    df_segments = df_segments.drop(columns=["start", "end", "duration"])
    df_segments["duration_segment"] = df_segments["end_segment"] - df_segments["start_segment"]

    # Group by and split the dataframe into list of dfs based on the audio_file column.
    # Only done to parallelize the process.
    df_fuzzy_list = df_fuzzy.groupby("audio_file", as_index=False, group_keys=False)
    df_fuzzy_list = [df for _, df in df_fuzzy_list]

    # Process in parallel
    with mp.Pool(mp.cpu_count() - 2) as pool:
        min_overlap = 0.0000001
        args = [(df, df_segments, min_overlap) for df in df_fuzzy_list]
        df_overlap = list(
            tqdm(
                pool.imap(find_overlaps_multi, args, chunksize=20),
                total=len(df_fuzzy_list),
            )
        )

    df_overlap = pd.concat(df_overlap, ignore_index=True)

    # Calculate the fraction of each speaker's speech duration within each speech_id
    df_overlap = speaker_ratio_in_speeches(df_overlap)
    df_overlap["nr_speech_segments"] = df_overlap.groupby(["audio_file", "speech_id"])[
        "label"
    ].transform("count")

    # Filter out segments with low speaker duration ratio
    df_overlap = df_overlap[(df_overlap["speaker_ratio_normalized"] > 0.25)]
    df_overlap = df_overlap.sort_values(["audio_file", "speech_segment_id"]).reset_index(drop=True)

    # Subset first and last segment of a contiguous speech sequence by the same speaker in each speech_id
    df_overlap = subset_first_last_segment(df_overlap)

    # Start of contiguous speech segment
    df_overlap["start_segment"] = df_overlap.groupby(["audio_file", "speech_id", "label"])[
        "start_segment"
    ].transform("min")
    # End of speech segment
    df_overlap["end_segment"] = df_overlap.groupby(["audio_file", "speech_id", "label"])[
        "end_segment"
    ].transform("max")

    df_overlap["overall_score"] = (
        df_overlap["overlap_ratio"] + df_overlap["speaker_ratio_normalized"]
    ) / 2

    # Pick the best speaker (label) candidate speaker within each speech_id based on the overall_score
    df_overlap = (
        df_overlap.groupby(["audio_file", "label", "speech_id"])[df_overlap.columns.tolist()]
        .apply(lambda x: x.loc[x["overall_score"].idxmax()])
        .reset_index(drop=True)
    )

    # df_overlap = df_overlap.groupby(["audio_file", "label", "speech_id"]).first().reset_index()
    df_overlap["duration_segment"] = df_overlap["end_segment"] - df_overlap["start_segment"]

    df_overlap["duration_overlap"] = df_overlap[["end_segment", "end_text_time"]].min(
        axis=1
    ) - df_overlap[["start_segment", "start_text_time"]].max(axis=1)
    df_overlap["duration_text"] = df_overlap["end_text_time"] - df_overlap["start_text_time"]
    # Recall-ish
    df_overlap["overlap_ratio"] = df_overlap["duration_overlap"] / df_overlap["duration_text"]
    # Precision-ish
    df_overlap["length_ratio"] = df_overlap["duration_segment"] / df_overlap["duration_text"]
    df_overlap["length_ratio_normalized"] = df_overlap["length_ratio"].apply(normalize_above_1)

    df_overlap["overall_score"] = (
        df_overlap["overlap_ratio"]
        + df_overlap["length_ratio_normalized"]
        + df_overlap["speaker_ratio_normalized"]
    ) / 3

    # Keep only the segment with the highest overall_score within each speech_id
    df_overlap = (
        df_overlap.groupby(["audio_file", "speech_id"])[df_overlap.columns.tolist()]
        .apply(lambda x: x.loc[x["overall_score"].idxmax()])
        .reset_index(drop=True)
    )

    df_overlap = df_overlap[df_overlap["overall_score"] > 0.6].reset_index(drop=True)
    df_overlap.to_parquet(os.path.join(args.data_dir, "df_overlap.parquet"), index=False)

    df_fuzzy["text"] = df_fuzzy["anf_text"]
    df_fuzzy["date"] = pd.to_datetime(df_fuzzy["anf_datum"])
    df_fuzzy["dead"] = None
    df_overlap = df_overlap.merge(
        df_fuzzy[["speech_id", "date", "name", "party", "gender", "text", "riksdagen_id", "born"]],
        on="speech_id",
        how="left",
    )

    df_overlap["start_segment_same"] = (
        df_overlap.groupby("audio_file")["start_segment"].diff() == 0
    ) | (df_overlap.groupby("audio_file")["start_segment"].diff(-1) == 0)

    df_overlap["person_id"] = df_overlap["speaker_id"]
    df_overlap = df_overlap[
        [
            "speech_id",
            "name",
            "party",
            "text",
            "text_normalized",
            "start_segment",
            "end_segment",
            "duration_segment",
            "start_text_time",
            "end_text_time",
            "nr_speech_segments",
            "start_segment_same",
            "overall_score",
            "speaker_id",
            "riksdagen_id",
            "protocol_id",
            "person_id",
            "born",
            "date",
            "audio_file",
        ]
    ]

    df_overlap.to_parquet(os.path.join(args.data_dir, "rixvox-alignments.parquet"), index=False)
