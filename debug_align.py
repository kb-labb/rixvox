import os
import re
from typing import List

import ctc_segmentation
import numpy as np
import pandas as pd
import pysrt
import simplejson as json
import torch
import torchaudio
import torchaudio.functional as F
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor

from rixvox.text import normalize_text


def align_pytorch(transcripts, emissions, device):
    transcript = " ".join(transcripts)
    transcript = transcript.replace("\n", " ").upper()
    targets = processor.tokenizer(transcript, return_tensors="pt")["input_ids"]
    targets = targets.to(device)

    alignments, scores = F.forced_align(emissions, targets, blank=0)

    alignments, scores = alignments[0], scores[0]  # remove batch dimension for simplicity
    # scores = scores.exp()  # convert back to probability
    return alignments, scores


def align_with_transcript(
    transcripts: List[str],
    probs: torch.Tensor,
    audio_frames: int,
    processor: Wav2Vec2Processor,
    samplerate: int = 16000,
):
    # Tokenize transcripts
    vocab = processor.tokenizer.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    unk_id = vocab["<unk>"]

    tokens = []
    for transcript in transcripts:
        assert len(transcript) > 0
        tok_ids = processor.tokenizer(transcript.replace("\n", " ").upper())["input_ids"]
        tok_ids = np.array(tok_ids, dtype="int")
        tokens.append(tok_ids[tok_ids != unk_id])

    probs = probs[0].cpu().numpy() if probs.ndim == 3 else probs.cpu().numpy()
    # Get nr of encoded CTC frames in the encoder without padding.
    # I.e. the number of "tokens" the audio was encoded into.
    ctc_frames = calculate_w2v_output_length(audio_frames, chunk_size=30)

    # Align
    char_list = [inv_vocab[i] for i in range(len(inv_vocab))]
    config = ctc_segmentation.CtcSegmentationParameters(char_list=char_list)
    config.index_duration = audio_frames / ctc_frames / samplerate
    print(f"Index duration: {config.index_duration}")
    print(f"audio_frames: {audio_frames}")
    print(f"ctc_frames: {ctc_frames}")
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


def split_speech_from_media(row, audiofile):
    start_speech = row["start_segment"]
    end_speech = row["end_segment"]
    speech_id = row["speech_id"]

    # Extract the audio from the start to the end of the speech with ffmpeg
    os.makedirs("data/tempaudio", exist_ok=True)
    audiofile = os.path.join("/data/faton/riksdagen_old/all", audiofile.rsplit("/")[-1])
    basename = os.path.basename(audiofile)
    speech_audiofile = os.path.join("data/tempaudio", f"{basename}_{speech_id}.wav")

    # Convert the video to wav 16kHz mono from the start to the end of the speech
    os.system(
        f"ffmpeg -i {audiofile} -ac 1 -ar 16000 -ss {start_speech} -to {end_speech} {speech_audiofile}"
    )

    return {
        "speech_audiofile": speech_audiofile,
        "start_speech": start_speech,
        "end_speech": end_speech,
    }


def get_probs(speech_metadata, pad=False, logits_only=False):
    # Load the audio file
    speech_audiofile = speech_metadata["speech_audiofile"]
    audio_input, sr = torchaudio.load(speech_audiofile)
    audio_input.to(device).half()  # Convert to half precision

    # Split the audio into chunks of 30 seconds
    chunk_size = 20
    audio_chunks = torch.split(audio_input, chunk_size * sr, dim=1)

    # Transcribe each audio chunk
    all_probs = []

    for audio_chunk in audio_chunks:
        # If audio chunk is shorter than 30 seconds, pad it to 30 seconds
        if audio_chunk.shape[1] < chunk_size * sr:
            padding = torch.zeros((1, chunk_size * sr - audio_chunk.shape[1]))
            audio_chunk = torch.cat([audio_chunk, padding], dim=1)
        input_values = (
            processor(audio_chunk, sampling_rate=16000, return_tensors="pt", padding="longest")
            .input_values.to(device)
            .squeeze(dim=0)
        )
        with torch.inference_mode():
            logits = model(input_values.half()).logits
            if logits_only:
                probs = logits
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1)

        all_probs.append(probs)

    # Concatenate the probabilities
    align_probs = torch.cat(all_probs, dim=1)
    return align_probs, len(audio_input[0])


def is_only_non_alphanumeric(text):
    """
    re.match returns a match object if the pattern is found and None otherwise.
    """
    # Contains only 1 or more non-alphanumeric characters
    return re.match(r"^[^a-zA-Z0-9]+$", text) is not None


def word_tokenize(text):
    text = row["anf_text"].split(" ")  # word tokenization
    text = [token for token in text if is_only_non_alphanumeric(token) is False]
    return text


def format_timestamp(timestamp):
    """
    Convert timestamp in seconds to "hh:mm:ss,ms" format
    expected by pysrt.
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

    Parameters
    ----------
    audio_frames
        Number of audio frames in the audio file, or part of audio file to be aligned.
    chunk_size
        Number of seconds to chunk the audio by for batched inference.
    conv_stride
        The convolutional stride of the wav2vec2 model (see model.config.conv_stride).
        The product sum of the list is the number of audio frames per output logit.
        Defaults to the conv_stride of wav2vec2-large.
    sample_rate
        The sample rate of the w2v processor, default 16000.
    frames_first_logit
        First logit consists of more frames than the rest. Wav2vec2-large outputs
        the first logit after 400 frames.
    """
    frames_per_logit = np.prod(conv_stride)
    extra_frames = frames_first_logit - frames_per_logit

    frames_per_full_chunk = chunk_size * sample_rate
    n_full_chunks = audio_frames // frames_per_full_chunk

    # Calculate the number of logit outputs for the full size chunks
    logits_per_full_chunk = (frames_per_full_chunk - extra_frames) // frames_per_logit
    n_full_chunk_logits = n_full_chunks * logits_per_full_chunk

    # Calculate the number of logit outputs for the last chunk (may be shorter than the chunk size)
    n_last_chunk_frames = audio_frames % frames_per_full_chunk

    if n_last_chunk_frames == 0:
        n_last_chunk_logits = 0
    elif n_last_chunk_frames < frames_first_logit:
        # We'll pad the last chunk to 400 frames if it's shorter than the model's minimum input length
        n_last_chunk_logits = 1
    else:
        n_last_chunk_logits = (n_last_chunk_frames - extra_frames) // frames_per_logit

    return n_full_chunk_logits + n_last_chunk_logits


def unflatten(list_, lengths):
    assert len(list_) == sum(lengths)
    i = 0
    ret = []
    for l in lengths:
        ret.append(list_[i : i + l])
        i += l
    return ret


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCTC.from_pretrained(
        "KBLab/wav2vec2-large-voxrex-swedish", torch_dtype=torch.float16
    ).to(device)
    processor = Wav2Vec2Processor.from_pretrained(
        "KBLab/wav2vec2-large-voxrex-swedish", sample_rate=16000, return_tensors="pt"
    )

    # Get metadata for a riksdag debate through the API for a given debate document id
    with open(
        "/home/fatrek/data_network/delat/audio/riksdagen/data/speeches_by_audiofile/RD_EN_L_1978-12-18_1978-12-19.1.json"
    ) as f:
        meta = json.load(f)

    speeches_metadata = []
    for i, row in enumerate(meta["speeches"]):
        # Create a wav file with only the speech's audio
        audio_speech = split_speech_from_media(row, audiofile=meta["metadata"]["audio_file"])
        speeches_metadata.append(audio_speech)

    align_probs = []
    audio_frames = []
    for speech_metadata in tqdm(speeches_metadata):
        # Run transcription but only keep the probabilities for alignment
        probs, audio_length = get_probs(speech_metadata, pad=True, logits_only=False)
        align_probs.append(probs)
        audio_frames.append(audio_length)

    normalized_transcripts = []
    original_transcripts = []
    for i, row in enumerate(meta["speeches"]):
        # Chunk text according to what granularity we want alignment timestamps.
        # We sentence tokenize here, but we could also word tokenize, and then
        # at a later stage decide how to create subtitle chunks from word timestamps.

        transcript = sent_tokenize(row["text"])
        # transcript = word_tokenize(row["anf_text"])
        normalized_transcript = [normalize_text(token).upper() for token in transcript]
        normalized_transcripts.append(normalized_transcript)
        original_transcripts.append(transcript)

    alignments = []
    alignment_scores = []
    aligned_tokens = []
    for i, speech_metadata in enumerate(tqdm(speeches_metadata)):
        # # Alignment of audio with transcript
        # align = align_with_transcript(
        #     normalized_transcripts[i],
        #     align_probs[i],
        #     audio_frames[i],
        #     processor,
        # )
        # # for segment in align:
        # #     segment["start"] += float(speech_metadata["start_speech"])
        # #     segment["end"] += float(speech_metadata["start_speech"])

        tokens, scores = align_pytorch(normalized_transcripts[i], align_probs[i], device)
        # alignments.append(aligned_tokens)
        alignment_scores.append(scores)
        aligned_tokens.append(tokens)

    token_spans = F.merge_tokens(aligned_tokens[18], alignment_scores[18], blank=0)
    transcript = " ".join(normalized_transcripts[18])
    transcript = transcript.split()
    # Remove all TokenSpan with token=4 (| space)
    token_spans = [s for s in token_spans if s.token != 4]
    word_spans = unflatten(token_spans, [len(word) for word in transcript])
    calculate_w2v_output_length(audio_frames[18], chunk_size=20)
    len(alignment_scores[18])
    ratio = audio_frames[18] / calculate_w2v_output_length(audio_frames[18], chunk_size=20)

    def get_word_timing(original_word, word_span, ratio, sample_rate=16000):
        start = (word_span[0].start * ratio) / sample_rate
        end = (word_span[-1].end * ratio) / sample_rate
        print(f"{start} - {end}: {original_word}")

    for word_span, original_word in zip(word_spans[2000:2200], transcript[2000:2200]):
        get_word_timing(original_word, word_span, ratio)

    for s in token_spans:
        # Decode
        decoded = processor.decode(s.token)
        print(f"{s.start} - {s.end} - {s.score}: {decoded}")

    for alignment in alignments[18]:
        decoded = processor.decode(alignment)
        print(decoded)

    # Flatten the alignments
    alignments = [segment for speech in alignments for segment in speech]
    # Flatten the original transcripts
    transcripts = [token for speech in original_transcripts for token in speech]

    # Create a subtitles file from the timestamps
    subs = pysrt.SubRipFile()
