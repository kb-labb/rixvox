import itertools

import ctc_segmentation
import numpy as np
import torch
from transformers import Wav2Vec2Processor

# load dummy dataset and read soundfiles
SAMPLERATE = 16000


def align_with_transcript(
    transcripts: list[str],
    probs: np.ndarray,
    audio_frames: int,
    processor: Wav2Vec2Processor,
    samplerate: int = 16000,
    chunk_size: int = 30,
):
    """
    Get alignment timestamps for each each word/sentence in a "ground truth" transcript.

    Adapted from Wav2vec2 example code: https://github.com/lumaku/ctc-segmentation?tab=readme-ov-file#usage

    Args:
        transcripts: transcript organized as a list of words or sentences
            (or other tokenization unit) to align.
        probs: model output probabilities for the relevant segment of audio.
            Shape: (batch_size, seq_len, vocab_size) or (seq_len, vocab_size).
        audio_frames: number of audio frames in the segment of audio.
        processor: Wav2Vec2Processor object containing the tokenizer.
        samplerate: sample rate of the audio.
    """
    # Tokenize transcripts
    vocab = processor.tokenizer.get_vocab()
    char_list = list(vocab.keys())
    unk_id = vocab["<unk>"]

    tokens = []
    for transcript in transcripts:
        assert len(transcript) > 0
        tok_ids = processor.tokenizer(transcript.replace("\n", " ").upper())["input_ids"]
        tok_ids = np.array(tok_ids, dtype="int")
        tokens.append(tok_ids[tok_ids != unk_id])

    probs = np.concatenate(probs, axis=0) if probs.ndim == 3 else probs
    # Get nr of logits in the encoder (without padding added).
    # I.e. the number of "tokens" the audio was encoded into, or the number of
    # "character" predictions the model will output.
    nr_ctc_logits = calculate_w2v_output_length(audio_frames, chunk_size=chunk_size)

    # Align
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio_frames / nr_ctc_logits / samplerate
    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_token_list(config, tokens)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
        config, probs, ground_truth_mat
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
    chunk_size: int = 30,
) -> list[dict]:
    """
    Get timestamps for each word in the wav2vec2 model's prediction.

    Adapted from Wav2vec2 example code: https://github.com/lumaku/ctc-segmentation?tab=readme-ov-file#usage
    """
    # Split the transcription into words
    pred_transcript = pred_transcript.upper()  # vocab is upper case
    words = pred_transcript.split(" ")

    probs = probs[0].cpu().numpy() if probs.ndim == 3 else probs.cpu().numpy()
    # Get nr of logits in the encoder (without padding added).
    # I.e. the number of "tokens" the audio was encoded into, or the number of
    # "character" predictions the model will output.
    nr_ctc_logits = calculate_w2v_output_length(audio_frames, chunk_size=chunk_size)

    # Align
    vocab = processor.tokenizer.get_vocab()
    char_list = list(vocab.keys())
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio_frames / nr_ctc_logits / samplerate

    ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(config, words)
    timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
        config, probs, ground_truth_mat
    )
    segments = ctc_segmentation.determine_utterance_segments(
        config, utt_begin_indices, char_probs, timings, words
    )
    return [{"text": w, "start": p[0], "end": p[1], "conf": p[2]} for w, p in zip(words, segments)]


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
