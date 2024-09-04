import os
import re
from typing import List

import ctc_segmentation
import numpy as np
import pandas as pd
import simplejson as json
import torch
import torchaudio
import torchaudio.functional as F
from nltk.data import load
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktParameters
from nltk.tokenize.punkt import PunktSentenceTokenizer as pt
from tqdm import tqdm
from transformers import AutoModelForCTC, Wav2Vec2Processor

from rixvox.text import get_normalized_tokens, normalize_text_with_mapping


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

    mappings = []
    for i, row in enumerate(meta["speeches"]):
        normalized_text, mapping, original_text = normalize_text_with_mapping(row["text"])
        normalized_mapping, normalized_tokens = get_normalized_tokens(mapping)
        mappings.append(
            {
                "original_text": original_text,
                "mapping": mapping,
                "normalized_mapping": normalized_mapping,
                "normalized_tokens": normalized_tokens,
            }
        )

    # Align with pytorch
    alignments = []
    alignment_scores = []
    for i, speech_metadata in enumerate(tqdm(speeches_metadata)):
        tokens, scores = align_pytorch(mappings[i]["normalized_tokens"], align_probs[i], device)
        alignments.append(tokens)
        alignment_scores.append(scores)

    for i, speech_metadata in enumerate(tqdm(speeches_metadata)):
        token_spans = F.merge_tokens(alignments[i], alignment_scores[i], blank=0)
        # Remove all TokenSpan with token=4 (| space)
        token_spans = [s for s in token_spans if s.token != 4]
        word_spans = unflatten(
            token_spans, [len(word) for word in mappings[i]["normalized_tokens"]]
        )
        ratio = audio_frames[i] / calculate_w2v_output_length(audio_frames[i], chunk_size=20)

        multi_word = []
        for aligned_token, normalized_token in zip(
            word_spans, mappings[i]["normalized_mapping"].items()
        ):
            original_index = normalized_token[1]["normalized_word_index"]
            original_token = mappings[i]["mapping"][original_index]
            start_time, end_time = get_word_timing(
                aligned_token, ratio, start_segment=speech_metadata["start_speech"]
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

    # Recreate the original text

    tokenizer = load("tokenizers/punkt/swedish.pickle")

    sentence_spans = tokenizer.span_tokenize(mappings[18]["original_text"])
    sentence_mapping = []
    word_mapping = mappings[18]["mapping"].copy()
    for span in sentence_spans:
        start_sentence_index = span[0]  # Character index in the original text
        end_sentence_index = span[1]
        index = 0
        while word_mapping:
            word = word_mapping[0]

            # Print debug variables
            print(
                f"{start_sentence_index} - {end_sentence_index} - {word['original_start']} - {word['original_end']} - {word['original_token']}"
            )
            if start_sentence_index in list(range(word["original_start"], word["original_end"])):
                start_sentence_time = word["start_time"]

            elif (end_sentence_index - 1) in list(
                range(word["original_start"], word["original_end"])
            ):
                if word["end_time"] is None:
                    end_sentence_time = previous_removed["end_time"]
                else:
                    end_sentence_time = word["end_time"]
                if start_sentence_time is None:
                    try:
                        start_sentence_time = previous_removed["end_time"]
                    except NameError:
                        start_sentence_time = None

                sentence_mapping.append(
                    {
                        "start_sentence": start_sentence_time,
                        "end_sentence": end_sentence_time,
                        "text": mappings[18]["original_text"][
                            start_sentence_index:end_sentence_index
                        ],
                    }
                )
                break

            previous_removed = word_mapping.pop(0)

    # Reconstruct original tokens from mapping
    original_tokens = []
    for transformation in mappings[0]["mapping"]:
        original_tokens.append(transformation["original_token"])

    list(tokenizer.sentences_from_tokens(original_tokens))
    list(
        tokenizer.span_tokenize(
            "Hej, jag heter Nils. Jag är t.ex. en människa  . Det är bl.a. så att jag heter Nils. \n Det är bl. a. så."
        )
    )

    for word in mappings[18]["normalized_mapping"].items():
        print(f"{word[1]['start_time']} - {word[1]['end_time']}: {word[1]['token']}")

    mappings[18]["normalized_mapping"].items()[0:100]

    sent_tokenize(
        "Hej, jag heter Nils. Jag är t.ex. en människa. Det är bl.a. så att jag heter Nils. Det är bl. a. så.",
        language="swedish",
    )
    spans = list(
        pt().span_tokenize(
            "Hej, jag heter Nils. Jag är t.ex. en människa. Det är bl.a. så att jag heter Nils. Det är bl. a. så."
        )
    )
