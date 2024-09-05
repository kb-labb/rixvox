import ctc_segmentation
import numpy as np
import torch
from transformers import Wav2Vec2Processor

from rixvox.alignment import calculate_w2v_output_length


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
