from typing import List

import ctc_segmentation
import numpy as np
import torch
from transformers import Wav2Vec2Processor

# load dummy dataset and read soundfiles
SAMPLERATE = 16000


def align_with_transcript(
    transcripts: List[str],
    probs: torch.Tensor,
    audio_frames: int,
    processor: Wav2Vec2Processor,
    samplerate: int = SAMPLERATE,
):
    # Tokenize transcripts
    vocab = processor.tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    unk_id = vocab["<unk>"]

    tokens = []
    for transcript in transcripts:
        assert len(transcript) > 0
        tok_ids = processor.tokenizer(transcript.replace("\n", " ").upper())["input_ids"]
        tok_ids = np.array(tok_ids, dtype=np.int)
        tokens.append(tok_ids[tok_ids != unk_id])

    # Get nr of characters in the model output (if batched, it's the second dimension)
    probs = probs.cpu().numpy()
    probs_size = probs.shape[1] if probs.ndim == 3 else probs.shape[0]

    # Align
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio_frames / probs.size()[0] / samplerate

    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
        config, probs_size, ground_truth_mat
    )
    segments = ctc_segmentation.determine_utterance_segments(
        config, utt_begin_indices, char_probs, timings, transcripts
    )
    return [
        {"text": t, "start": p[0], "end": p[1], "conf": p[2]}
        for t, p in zip(transcripts, segments)
    ]


def get_word_timestamps(
    pred_transcript: str,
    probs: torch.Tensor,
    audio_frames: int,
    processor: Wav2Vec2Processor,
    samplerate: int = SAMPLERATE,
):
    """
    Adapted from Wav2vec2 example code: https://github.com/lumaku/ctc-segmentation?tab=readme-ov-file#usage
    """
    # Split the transcription into words
    pred_transcript = pred_transcript.upper()  # vocab is upper case
    words = pred_transcript.split(" ")

    # Get nr of characters in the model output (if batched, it's the second dimension)
    probs = probs.cpu().numpy()
    probs_size = probs.shape[1] if probs.ndim == 3 else probs.shape[0]

    # Align
    vocab = processor.tokenizer.get_vocab()
    char_list = list(vocab.keys())
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio_frames / probs_size / samplerate

    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, words)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
        config, probs, ground_truth_mat
    )
    segments = ctc_segmentation.determine_utterance_segments(
        config, utt_begin_indices, char_probs, timings, words
    )
    return [{"text": w, "start": p[0], "end": p[1], "conf": p[2]} for w, p in zip(words, segments)]


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
