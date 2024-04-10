import json

from transformers import pipeline

from rixvox.parlaspeech.matching import Matcher, load_segments, load_valid_json
from subtitles_riksdag import get_audio_metadata

audiofile = "data/audio/2442404050040696921_720p.mp4"

# Concatenate all texts and write to sample.txt
meta = get_audio_metadata("hb10625")
all_text = meta["anftext"].str.cat(sep=" ")

with open("data/sample.txt", "w") as f:
    f.write(all_text)

# Load the model
pipe = pipeline(
    "automatic-speech-recognition", model="KBLab/wav2vec2-large-voxrex-swedish", device="cuda"
)

# Transcribe the audio
transcript = pipe(audiofile, chunk_length_s=30, return_timestamps="word")

# An entry in chunks looks like this: {'text': 'PÅ', 'timestamp': (2014.28, 2014.32)}
# Combine entries up to 30 seconds in length
transcript_chunks = []
chunk = ""
start = 0
end = 0

for entry in transcript["chunks"]:
    chunk += entry["text"] + " "
    end = entry["timestamp"][1]

    if end - start >= 25:
        transcript_chunks.append(
            {
                "file": audiofile,
                "start": start,
                "end": end,
                "text": chunk,
            }
        )
        chunk = ""
        start = end

# Add the last chunk
transcript_chunks.append(
    {
        "file": audiofile,
        "start": start,
        "end": end,
        "text": chunk,
    }
)

# Write transcript to asr_results.json
with open("data/asr_results.json", "w") as f:
    json.dump(transcript_chunks, f, indent=4, ensure_ascii=False)

# Write the above as jsonl file
with open("data/asr_results.jsonl", "w") as f:
    for chunk in transcript_chunks:
        f.write(json.dumps(chunk) + "\n")

matcher = Matcher("data/sample.txt", normalize=True)

matcher.vocab
asr_results = load_segments("data/asr_results.jsonl")
positions = matcher.match(asr_results)

matcher.print_debug(positions)
positions = matcher.resegment_positions(positions)
matcher.print_debug(positions)
