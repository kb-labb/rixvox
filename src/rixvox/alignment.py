import itertools
import logging
import os

import numpy as np
import torch
import torchaudio.functional as F
from nltk.tokenize.punkt import PunktSentenceTokenizer
from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor

from rixvox.text import get_normalized_tokens, normalize_text_with_mapping

logger = logging.getLogger(__name__)


def align_pytorch(transcripts, emissions, processor, device):
    transcript = " ".join(transcripts)
    transcript = transcript.replace("\n", " ").upper()
    targets = processor.tokenizer(transcript, return_tensors="pt")["input_ids"]
    targets = targets.to(device)

    alignments, scores = F.forced_align(emissions, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    # scores = scores.exp()  # convert back to probability
    return alignments, scores


def format_timestamp(timestamp):
    """
    Convert timestamp in seconds to "hh:mm:ss:ms" format.
    """
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) // 60)
    seconds = int(timestamp % 60)
    milliseconds = int((timestamp % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def calculate_w2v_output_length(
    audio_frames: int,
    chunk_size: int,
    conv_stride: list[int] = [5, 2, 2, 2, 2, 2, 2],
    sample_rate: int = 16000,
    frames_first_logit: int = 400,
):
    """
    Calculate the number of output characters from the wav2vec2 model based
    on the chunking strategy and the number of audio frames.

    The wav2vec2-large model outputs one logit per 320 audio frames. The exception
    is the first logit, which is output after 400 audio frames (the model's minimum
    input length).

    We need to take into account the first logit, otherwise the alignment will slowly
    drift over time for long audio files when chunking the audio for batched inference.

    Args:
        audio_frames:
            Number of audio frames in the audio file, or part of audio file to be aligned.
        chunk_size:
            Number of seconds to chunk the audio by for batched inference.
        conv_stride:
            The convolutional stride of the wav2vec2 model (see model.config.conv_stride).
            The product sum of the list is the number of audio frames per output logit.
            Defaults to the conv_stride of wav2vec2-large.
        sample_rate:
            The sample rate of the w2v processor, default 16000.
        frames_first_logit:
            First logit consists of more frames than the rest. Wav2vec2-large outputs
            the first logit after 400 frames.

    Returns:
        The number of logit outputs for the audio file.
    """
    frames_per_logit = np.prod(conv_stride)  # 320 for wav2vec2-large
    extra_frames = frames_first_logit - frames_per_logit

    frames_per_full_chunk = chunk_size * sample_rate  # total frames for chunk_size seconds
    n_full_chunks = audio_frames // frames_per_full_chunk

    # Calculate the number of logit outputs for the full size chunks
    logits_per_full_chunk = (frames_per_full_chunk - extra_frames) // frames_per_logit
    n_full_chunk_logits = n_full_chunks * logits_per_full_chunk

    # Calculate the number of logit outputs for the last chunk (may be shorter than the chunk size)
    n_last_chunk_frames = audio_frames % frames_per_full_chunk

    if n_last_chunk_frames == 0:
        n_last_chunk_logits = 0
    elif n_last_chunk_frames < frames_first_logit:
        # We'll pad the last chunk up to 400 frames if it happens to be shorter
        # than the model's minimum input length (otherwise model will throw an error).
        n_last_chunk_logits = 1
    else:
        n_last_chunk_logits = (n_last_chunk_frames - extra_frames) // frames_per_logit

    return n_full_chunk_logits + n_last_chunk_logits


def segment_speech_probs(probs_list: list[np.ndarray], speech_ids: list[str]):
    """
    Divide the accumulated probs of audio file into the speeches they belong to.
    (we can't assume that a batch maps to a single speech)

    Args:
        probs_list: List of np.ndarrays containing the probs
            with shape (batch_size, seq_len, vocab_size).
        speech_ids: List of speech ids that each chunk (observation)
            in the probs_list belongs to.
    """
    speech_chunk_counts = [
        (key, sum(1 for i in group)) for key, group in itertools.groupby(speech_ids)
    ]
    split_indices = list(itertools.accumulate([count for _, count in speech_chunk_counts]))[:-1]

    probs_in_speech = np.concatenate(probs_list, axis=0)
    probs_split = np.split(probs_in_speech, split_indices, axis=0)
    unique_speech_ids = dict.fromkeys(speech_ids).keys()  # Preserves order

    assert len(speech_chunk_counts) == len(probs_split) == len(set(unique_speech_ids))
    for speech_id, probs in zip(unique_speech_ids, probs_split):
        yield speech_id, probs


def map_text_to_tokens(json_dict: dict) -> list[dict]:
    """
    Map text to normalized tokens, keeping track of character/token indices
    in the original text.
    """
    speeches = json_dict["speeches"]
    mappings = []
    for i, speech in enumerate(speeches):
        normalized_text, mapping, original_text = normalize_text_with_mapping(speech["text"])
        normalized_mapping, normalized_tokens = get_normalized_tokens(mapping)
        speech["text_normalized"] = normalized_text
        mappings.append(
            {
                "original_text": original_text,
                "mapping": mapping,
                "normalized_mapping": normalized_mapping,
                "normalized_tokens": normalized_tokens,
            }
        )

    return mappings, json_dict


def get_alignments_and_scores(
    json_dict: dict,
    mappings: list[dict],
    processor: Wav2Vec2Processor,
    probs_dir: str,
    device: str,
) -> tuple:
    """
    Get the alignments and scores for the tokens in the normalized text.
    """
    speeches = json_dict["speeches"]
    alignments = []
    alignment_scores = []
    for i, mapping in enumerate(mappings):
        align_probs = np.load(
            os.path.join(probs_dir, speeches[i]["probs_file"]), allow_pickle=True
        )
        # align_probs have dim (batches, nr_logits, vocab_size),
        # We want to stack to (1, batch_size * nr_logits, vocab_size)
        align_probs = np.vstack(align_probs)
        align_probs = torch.tensor(align_probs, device=device).unsqueeze(0)

        try:
            tokens, scores = align_pytorch(
                mapping["normalized_tokens"], align_probs, processor, device
            )
        except Exception as e:
            logger.error(
                f"Failed to align speech {speeches[i]['speech_id']} from file {json_dict['metadata']['audio_file']}: {e}"
            )
            tokens = []
            scores = []
        alignments.append(tokens)
        alignment_scores.append(scores)

    return alignments, alignment_scores


def add_timestamps_to_mapping(
    json_dict: dict,
    mappings: list[dict],
    alignments: list[torch.Tensor],
    alignment_scores: list[torch.Tensor],
    chunk_size: int = 30,
) -> list[dict]:
    """
    Add the timestamps from aligned tokens to the original text tokens via the mapping.
    """
    speeches = json_dict["speeches"]
    for i, speech in enumerate(speeches):
        if (len(alignments[i]) == 0) or (len(alignment_scores[i]) == 0):
            continue

        token_spans = F.merge_tokens(alignments[i], alignment_scores[i], blank=0)
        # Remove all TokenSpan with token=4 (token 4 is "|", used for space)
        token_spans = [s for s in token_spans if s.token != 4]
        word_spans = unflatten(
            token_spans, [len(word) for word in mappings[i]["normalized_tokens"]]
        )
        ratio = speech["audio_frames"] / calculate_w2v_output_length(
            speech["audio_frames"], chunk_size=chunk_size
        )

        for aligned_token, normalized_token in zip(
            word_spans, mappings[i]["normalized_mapping"].items()
        ):
            original_index = normalized_token[1]["normalized_word_index"]
            original_token = mappings[i]["mapping"][original_index]
            start_time, end_time = get_word_timing(
                aligned_token, ratio, start_segment=speech["start_segment"]
            )

            if not normalized_token[1]["is_multi_word"]:
                normalized_token[1]["start_time"] = start_time
                normalized_token[1]["end_time"] = end_time
                original_token["start_time"] = start_time
                original_token["end_time"] = end_time
            else:
                if normalized_token[1]["is_first_word"]:
                    original_token["start_time"] = start_time
                if normalized_token[1]["is_last_word"]:
                    original_token["end_time"] = end_time

                normalized_token[1]["start_time"] = start_time
                normalized_token[1]["end_time"] = end_time

    return mappings


def get_sentence_alignment(
    json_dict: dict, mappings: list[dict], tokenizer: PunktSentenceTokenizer
):
    """
    Add the timestamps from aligned tokens to the sentenced tokenized original text.
    """
    speeches = json_dict["speeches"]
    for i, speech in enumerate(speeches):
        sentence_spans = tokenizer.span_tokenize(mappings[i]["original_text"])
        word_mapping = mappings[i]["mapping"].copy()
        if "start_time" not in word_mapping[0]:
            logger.info(
                f"Skipping {speech['speech_id']} in {json_dict['metadata']['audio_file']} because alignment is missing/failed."
            )
            speech["alignment"] = []
            continue

        sentence_mapping = []
        previous_removed = []
        for span in sentence_spans:
            start_sentence_index = span[0]  # Character index in the original text
            end_sentence_index = span[1]
            start_sentence_time = None
            end_sentence_time = None
            while word_mapping:
                word = word_mapping[0]

                if start_sentence_index in list(
                    range(word["original_start"], word["original_end"])
                ):
                    if word["start_time"] is None:
                        index = 0
                        start_sentence_time = None
                        while start_sentence_time is None:
                            # When start_time is None for a token, we search for a timestamp in
                            # following words that are still in stack
                            try:
                                start_sentence_time = word_mapping[index]["start_time"]
                                index += 1
                            except IndexError:
                                break
                    else:
                        start_sentence_time = word["start_time"]

                elif (end_sentence_index - 1) in list(
                    range(word["original_start"], word["original_end"])
                ):
                    if word["end_time"] is None:
                        index = -1
                        end_sentence_time = None
                        while end_sentence_time is None:
                            # When end_time is None for a token, we search for a timestamp in
                            # previous words that were removed from stack
                            try:
                                end_sentence_time = previous_removed[index]["end_time"]
                                index -= 1
                            except IndexError:
                                break
                    else:
                        end_sentence_time = word["end_time"]

                    if start_sentence_time is None or end_sentence_time is None:
                        # Skip sentence if we can't find start or end time
                        logger.info(
                            (
                                f"start_sentence_time missing for {speech['speech_id']} in "
                                f"{json_dict['metadata']['audio_file']} for sentence: "
                                f"{mappings[i]['original_text'][start_sentence_index:end_sentence_index]}"
                                f"and word: {word}"
                            )
                        )
                        break

                    sentence_mapping.append(
                        {
                            "start": start_sentence_time,
                            "end": end_sentence_time,
                            "start_hhmmssms": format_timestamp(start_sentence_time),
                            "end_hhmmssms": format_timestamp(end_sentence_time),
                            "text": mappings[i]["original_text"][
                                start_sentence_index:end_sentence_index
                            ],
                        }
                    )
                    break

                previous_removed.append(word_mapping.pop(0))

            speech["alignment"] = sentence_mapping

    return json_dict


def unflatten(list_, lengths):
    """
    Unflatten a list of character output tokens from wav2vec2 into words.

    Args:
        list_:
            A list of character tokens.
        lengths:
            A list of lengths of the words (normalized tokens).
    """
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


def get_word_timing(word_span, ratio, start_segment=0, sample_rate=16000):
    """
    Calculate the start and end time of a word span in the original audio file.

    Args:
        word_span:
            A list of TokenSpan objects representing the word span timings in the
            aligned audio chunk.
        ratio:
            The number of audio frames per model output logit. This is the
            total number of frames in our audio chunk divided by the number of
            (non-padding) logit outputs for the chunk.
        start_segment:
            The start time of the speech segment in the original audio file.
        sample_rate:
            The sample rate of the audio file, default 16000.

    """
    start = (word_span[0].start * ratio) / sample_rate + start_segment
    end = (word_span[-1].end * ratio) / sample_rate + start_segment
    return start, end


def get_word_timestamps_hf(word_offsets, time_offset):
    word_timestamps = []
    for word_offset in word_offsets:
        word_offset = [
            {
                "word": w["word"],
                "start_time": round(w["start_offset"] * time_offset, 2),
                "end_time": round(w["end_offset"] * time_offset, 2),
            }
            for w in word_offset
        ]
        word_timestamps.append(word_offset)
    return word_timestamps
