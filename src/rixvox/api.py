import json
import os
import time

import requests
from tqdm import tqdm

from rixvox.text import preprocess_audio_metadata


def get_audio_metadata(rel_dok_id, backoff_factor=0.2):
    """
    Download metadata for anföranden (speeches) to find which ones have related
    media files at riksdagens öppna data. The anföranden which have a
    rel_dok_id tend to be the ones that have associated media files.

    Args:
        rel_dok_id (str): rel_dok_id for the session. Retrieved from text
            transcript files at https://data.riksdagen.se/data/anforanden/.
        backoff_factor (int): Slow down the request frequency if riksdagen's
            API rejects requests.

    Returns:
        dict: Nested metadata fields with transcribed texts, media file
            URLs and more.
    """

    api_url = f"https://data.riksdagen.se/dokumentstatus/{rel_dok_id}.json?utformat=json&utdata=debatt,media"

    for i in range(3):
        backoff_time = backoff_factor * (2**i)
        speech_metadata = requests.get(api_url)

        if speech_metadata.status_code == 200:

            try:
                speech_metadata = json.loads(speech_metadata.text)
            except Exception as e:
                print(f"JSON decoding failed for rel_dok_id {rel_dok_id}. \n")
                print(e)
                return None

            # Check if speech_metadata["dokumentstatus"]["debatt"]["anforande"] exists
            if "debatt" not in speech_metadata["dokumentstatus"]:
                return None

            if "webbmedia" not in speech_metadata["dokumentstatus"]:
                print(
                    f"rel_dok_id {rel_dok_id} has no media files (webbmedia).",
                    end="\r",
                    flush=True,
                )
                return

            df = preprocess_audio_metadata(speech_metadata)
            df["rel_dok_id"] = rel_dok_id
            return df

        else:
            print(
                f"""rel_dok_id {rel_dok_id} failed with code {speech_metadata.status_code}.
                Retry attempt {i}: Retrying in {backoff_time} seconds""",
                end="\r",
                flush=True,
            )

        time.sleep(backoff_time)


def get_media_file(audiofileurl, backoff_factor=0.2, progress_bar=False):
    """
    Download mp3/mp4 files from riksdagens öppna data.
    Endpoint https://data.riksdagen.se/dokumentstatus/{rel_dok_id}.json?utformat=json&utdata=debatt,media

    Args:
        audiofileurl (str): Download URL for the mp3 audio file.
            E.g: https://mhdownload.riksdagen.se/VOD1/PAL169/2442205160012270021_aud.mp3
        backoff_factor (int): Slow down the request frequency if riksdagen's
            API rejects requests.

    Returns
    """

    os.makedirs("data/audio", exist_ok=True)

    if progress_bar:
        get_media_file_pbar(audiofileurl, backoff_factor)
        return None

    for i in range(3):
        file_path = os.path.join("data", "audio", audiofileurl.rsplit("/")[-1])

        if os.path.exists(file_path):
            print(f"File {file_path} has already downloaded.", end="\r", flush=True)
            break

        backoff_time = backoff_factor * (2**i)
        print(f"Downloading {audiofileurl} to {file_path}", end="\r", flush=True)

        speeches_media = requests.get(audiofileurl)

        if speeches_media.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(speeches_media.content)
            return None
        else:
            print(
                f"audiofileurl {audiofileurl} failed with code {speeches_media.status_code}",
                end="\r",
                flush=True,
            )

        time.sleep(backoff_time)


def get_media_file_pbar(audiofileurl, backoff_factor=0.2):
    """
    Download mp3/mp4 files from riksdagens öppna data with progress bar.
    Endpoint https://data.riksdagen.se/dokumentstatus/{rel_dok_id}.json?utformat=json&utdata=debatt,media

    Args:
        audiofileurl (str): Download URL for the mp3 audio file.
            E.g: https://mhdownload.riksdagen.se/VOD1/PAL169/2442205160012270021_aud.mp3
        backoff_factor (int): Slow down the request frequency if riksdagen's
            API rejects requests.

    Returns
    """

    os.makedirs("data/audio", exist_ok=True)

    for i in range(3):
        file_path = os.path.join("data", "audio", audiofileurl.rsplit("/")[-1])

        if os.path.exists(file_path):
            print(f"File {file_path} has already downloaded.", end="\r", flush=True)
            break

        backoff_time = backoff_factor * (2**i)
        print(f"Downloading {audiofileurl} to {file_path}", end="\r", flush=True)

        speeches_media = requests.get(audiofileurl, stream=True)
        total_size = int(speeches_media.headers.get("content-length", 0))
        block_size = 1024

        if speeches_media.status_code == 200:
            with tqdm(total=total_size, unit="B", unit_scale=True, leave=False) as pbar:
                with open(file_path, "wb") as f:
                    for data in speeches_media.iter_content(block_size):
                        pbar.update(len(data))
                        f.write(data)
            return None
        else:
            print(
                f"audiofileurl {audiofileurl} failed with code {speeches_media.status_code}",
                end="\r",
                flush=True,
            )

        time.sleep(backoff_time)
