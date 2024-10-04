## Rixvox v2: an automatic speech recognition dataset for Swedish

Source code to create RixVox v2 based on the Swedish Parliament's (Riksdagen) media recordings and text protocols. The pipeline consists of modules to:  

1. locate speeches in audio based on text protocols.
2. enhance accuracy of start/end times for the speeches with diarization.
3. force align text protocol of speech with audio of speech to get sentence/word timestamps.
4. create audio chunks ready for ASR training (up to 30 seconds chunks).
5. assess the quality of chunks/alignments via machine transcription of chunks and BLEU/WER scores.

## Source data

The dataset is derived from two primary sources. 

* A hard drive of older recordings from 1966-2002 that KBLab received from Riksdagen. These recordings were previously digitized in collaboration with The National Library of Sweden. There are 6825 audio files, each about 3 to 5 hours in length. For this material we have no other metadata aside from the possible date(s) they were recorded.
* Riksdagen's Web TV with recordings from 2000-2024. The parliament has its own Web TV that uploads recordings of parliamentary sessions. The media files are accessible via an API at the endpoint: `https://data.riksdagen.se/dokumentstatus/{dok_id}.json?utformat=json&utdata=debatt,media`. Where `dok_id` is the document id of the debate. For reference, here is the debate with id [HA01KU20](https://data.riksdagen.se/dokumentstatus/HA01KU20.json?utformat=json&utdata=debatt,media). 

## Instructions


### Riksdagen old recordings 1966-2002

Download the text protocols and metadata about speakers/persons from SWERIK's repo. This project used [v.1.0.0 of the Swedish Parliament Corpus](https://github.com/swerik-project/the-swedish-parliament-corpus/releases/tag/v2024.04.26).

* [persons_v1.0.0.zip](https://github.com/swerik-project/the-swedish-parliament-corpus/releases/download/v2024.04.26/persons_v1.0.0.zip)
* [records_v1.0.0.zip](https://github.com/swerik-project/the-swedish-parliament-corpus/releases/download/v2024.04.26/records_v1.0.0.zip)

1. `scripts/riksdagen_old/riksdagen_corpus.py`

### Riksdagen web 2000-2024

1. Download the text protocols of speeches from Riksdagens open data: [`bash scripts/utils/download_modern_speeches.sh`](https://github.com/kb-labb/rixvox/blob/main/scripts/utils/download_modern_speeches.sh).
2. Preprocess the text protocols: [`python scripts/riksdagen_web/preprocess_speech_metadata.py](https://github.com/kb-labb/rixvox/blob/main/scripts/riksdagen_web/preprocess_speech_metadata.py)
3. Download metadata about media recordings based on the document ids of text protocols: [`python scripts/riksdagen_web/download_audio_metadata.py`](https://github.com/kb-labb/rixvox/blob/main/scripts/riksdagen_web/download_audio_metadata.py), and join together this information with text protocol metadata.
4. Download the media files: [`python scripts/riksdagen_web/download_audio.py`](https://github.com/kb-labb/rixvox/blob/main/scripts/riksdagen_web/download_audio.py).
5. Perform fuzzy string matching between wav2vec2 machine transcription and text protocols to determine approximate start/end timestamp of each speech: [`python scripts/riksdagen_web/fuzzy_matcher.py](https://github.com/kb-labb/rixvox/blob/main/scripts/riksdagen_web/fuzzy_matcher.py).
6. Perform diarization on audio files to obtain more accurate start/end timestamp of each speech via speaker segments: [`python scripts/diarization_pyannote.py](https://github.com/kb-labb/rixvox/blob/main/scripts/diarization_pyannote.py). 
7. `scripts/diarization_preprocess.py`
8. `scripts/riksdagen_web/diarization_matcher.py`
9. `scripts/riksdagen_web/dataset_to_json.py`
10. `scripts/alignment_probs_writer.py`
11. `scripts/align_transcript_pytorch.py`
12. `scripts/create_chunks.py`
13. `scripts/lang_detect_whisper.py`
14. `scripts/transcribe_wav2vec2.py`
15. `scripts/transcribe_whisper.py`
16. `scripts/json_to_parquet.py`