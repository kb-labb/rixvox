SILENCE = "<|silence|>"


def n_non_silent_chunks(sub_dict) -> int:
    non_silent_chunks = [x for x in sub_dict["chunks"] if x["text"] != ""]
    return len(non_silent_chunks)


def seconds_to_ms(seconds):
    return round(seconds * 1_000)


def make_chunks(
    subs,
    min_threshold=10_000,
    max_threshold=30_000,
    surround_silence=True,
    silent_chunks=False,
):
    chunks = []
    chunk = []
    sub_ids = []
    total_length = 0
    chunk_start = 0
    chunk_end = 0
    speech_id = subs["speech_id"]

    # if "chunks" not in subs:
    #     subs["chunks"] = {}
    # if f"{min_threshold / 1_000}-{max_threshold / 1_000}" not in subs["chunks"]:
    #     subs["chunks"][f"{min_threshold / 1_000}-{max_threshold / 1_000}"] = chunks
    # else:
    #     raise Exception("Chunks with these thresholds already exist")

    def subs_to_whisper(subs):
        def whisper_time(ms):
            ms = ms // 20 * 20
            ms /= 1_000
            return f"<{ms:.2f}>"

        parts = []
        for sub in subs:

            if sub["text"] != SILENCE:
                part = "".join(
                    (
                        whisper_time(sub["start"]),
                        sub["text"].replace("\n", " "),
                        whisper_time(sub["end"]),
                    )
                )
                parts.append(part)
        return "".join(parts)

    def subs_to_raw_text(subs):
        return " ".join([sub["text"] for sub in subs if sub["text"].replace("\n", " ") != SILENCE])

    i = 0
    for sub_i, sub in enumerate(subs["alignment"]):
        sub["start"] = sub["start"]
        sub["end"] = sub["end"]
        sub["duration"] = sub["end"] - sub["start"]

        if max_threshold == 5_000:
            i = i + 1
        if i > 20:
            # don't make too many small chunks...
            i = 0
            break
        # silent subs get negative ids
        if sub["text"] == SILENCE:
            sub_i *= -1
        sub["id"] = sub_i
        sub = sub.copy()

        if sub["live"] or sub["duplicate"] or sub["is_long"]:
            if total_length >= min_threshold:
                chunks.append(
                    {
                        "start": chunk_start,
                        "end": chunk_end,
                        "duration": chunk_end - chunk_start,
                        "subs": chunk,
                        "text_whisper": subs_to_whisper(chunk),
                        "text": subs_to_raw_text(chunk),
                        "transcription": [],
                        "sub_ids": sub_ids,
                        "speech_id": speech_id,
                    }
                )
            # else: we throw away the chunk
            chunk = []
            sub_ids = []
            total_length = 0
            continue
        else:
            # add to chunk if total chunk length < 30s
            if sub["duration"] + total_length > max_threshold:
                if sub["text"] == SILENCE:
                    filler_silence = sub.copy()
                    filler_silence["duration"] = max_threshold - total_length
                    filler_silence["start"] = total_length
                    filler_silence["end"] = max_threshold
                    if surround_silence:
                        chunk_end = sub["start"] + filler_silence["duration"]
                        chunk.append(filler_silence)
                        sub_ids.append(sub_i)
                    chunks.append(
                        {
                            "start": chunk_start,
                            "end": chunk_end,
                            "duration": chunk_end - chunk_start,
                            "subs": chunk,
                            "text_whisper": subs_to_whisper(chunk),
                            "text": subs_to_raw_text(chunk),
                            "transcription": [],
                            "sub_ids": sub_ids,
                            "speech_id": speech_id,
                        }
                    )
                    chunk = []
                    sub_ids = []
                    total_length = 0
                    sub["start"] = sub["start"] + filler_silence["duration"]
                    sub["duration"] -= filler_silence["duration"]

                    while sub["duration"] > max_threshold:
                        chunk_start = sub["start"]
                        chunk_end = sub["start"] + max_threshold
                        if silent_chunks:
                            chunks.append(
                                {
                                    "start": chunk_start,
                                    "end": chunk_end,
                                    "duration": chunk_end - chunk_start,
                                    "subs": [
                                        {
                                            "text": SILENCE,
                                            "start": 0,
                                            "end": max_threshold,
                                            "duration": max_threshold,
                                        }
                                    ],
                                    "text_whisper": subs_to_whisper(chunk),
                                    "text": subs_to_raw_text(chunk),
                                    "transcription": [],
                                    "sub_ids": [sub["id"]],
                                    "speech_id": speech_id,
                                }
                            )
                        # chunk = []
                        # total_length = 0
                        sub["start"] += max_threshold
                        sub["duration"] -= max_threshold
                else:
                    chunks.append(
                        {
                            "start": chunk_start,
                            "end": chunk_end,
                            "duration": chunk_end - chunk_start,
                            "subs": chunk,
                            "text_whisper": subs_to_whisper(chunk),
                            "text": subs_to_raw_text(chunk),
                            "transcription": [],
                            "sub_ids": sub_ids,
                            "speech_id": speech_id,
                        }
                    )
                    chunk = []
                    sub_ids = []
                    total_length = 0

            # we either do not care about having silence at the beginning
            # or we are strict and start with a proper sub
            if surround_silence or (
                not surround_silence and len(chunk) == 0 and sub["text"] != SILENCE
            ):
                if len(chunk) == 0:
                    chunk_start = sub["start"]
                chunk_end = sub["end"]
                sub["start"] = total_length
                sub["end"] = total_length + sub["duration"]
                chunk.append(sub)
                sub_ids.append(sub_i)
                total_length += sub["duration"]

    if total_length >= min_threshold:
        chunks.append(
            {
                "start": chunk_start,
                "end": chunk_end,
                "duration": chunk_end - chunk_start,
                "subs": chunk,
                "text_whisper": subs_to_whisper(chunk),
                "text": subs_to_raw_text(chunk),
                "transcription": [],
                "sub_ids": sub_ids,
                "speech_id": speech_id,
            }
        )
    # TODO fix silent_chunk avoidance properly during creation
    if not silent_chunks:
        chunks = [chunk for chunk in chunks if not chunk["text"] == ""]

    if "chunks" not in subs:
        subs["chunks"] = chunks
    else:
        subs["chunks"].extend(chunks)

    return subs
