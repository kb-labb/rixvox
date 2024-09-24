import json
import logging
import os
import shutil
import time

import requests
from tqdm import tqdm

from rixvox.text import preprocess_audio_metadata

logger = logging.getLogger(__name__)


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
                logging.error(f"JSON decoding failed for rel_dok_id {rel_dok_id}. \n {e}")
                return None

            # Check if speech_metadata["dokumentstatus"]["debatt"]["anforande"] exists
            if "debatt" not in speech_metadata["dokumentstatus"]:
                logging.warning(f"rel_dok_id {rel_dok_id} has no debates (debatt).")
                return None

            if "webbmedia" not in speech_metadata["dokumentstatus"]:
                logging.warning(f"rel_dok_id {rel_dok_id} has no media files (webbmedia).")
                return None

            try:
                df = preprocess_audio_metadata(speech_metadata)
            except Exception as e:
                logging.error(f"Preprocessing failed for rel_dok_id {rel_dok_id}. \n {e}")
                return None
            df["rel_dok_id"] = rel_dok_id
            return df

        else:
            logging.info(
                f"rel_dok_id {rel_dok_id} failed with code {speech_metadata.status_code}. "
                f"Retry attempt {i}: Retrying in {backoff_time} seconds"
            )

        time.sleep(backoff_time)


def get_media_file(audiofileurl, backoff_factor=0.2, progress_bar=False, audio_dir="data/audio"):
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

    os.makedirs(audio_dir, exist_ok=True)

    if progress_bar:
        get_media_file_pbar(audiofileurl, backoff_factor)
        return None

    for i in range(3):
        file_path = os.path.join(audio_dir, audiofileurl.rsplit("/")[-1])

        if os.path.exists(file_path):
            logging.info(f"File {file_path} has already downloaded.")
            break

        backoff_time = backoff_factor * (2**i)
        logging.info(f"Downloading {audiofileurl} to {file_path}")

        speeches_media = requests.get(audiofileurl)

        if speeches_media.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(speeches_media.content)
            return None
        else:
            logging.info(
                f"audiofileurl {audiofileurl} failed with code {speeches_media.status_code}"
            )

        time.sleep(backoff_time)


def get_media_file_pbar(audiofileurl, backoff_factor=0.2, audio_dir="data/audio"):
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

    os.makedirs(audio_dir, exist_ok=True)

    for i in range(3):
        file_path = os.path.join(audio_dir, audiofileurl.rsplit("/")[-1])

        if os.path.exists(file_path):
            logging.info(f"File {file_path} has already downloaded.")
            break

        backoff_time = backoff_factor * (2**i)
        logging.info(f"Downloading {audiofileurl} to {file_path}")

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
            logging.info(
                f"audiofileurl {audiofileurl} failed with code {speeches_media.status_code}"
            )

        time.sleep(backoff_time)


def audio_to_dokid_folder(df, audio_dir="data/audio"):
    """
    Move audio files to a folder named after the dokid.

    Args:
        df (pd.DataFrame): A pandas dataframe with the relevant metadata fields.
    """

    # Only keep the first row for each dokid, they all have same audio file
    df = df.groupby("dok_id").first().reset_index()

    for _, row in tqdm(df.iterrows(), total=len(df)):
        dokid = row["dok_id"]
        audiofileurl = row["audiofileurl"]
        filename = audiofileurl.rsplit("/", 1)[1]
        src = os.path.join(audio_dir, filename)
        dst = os.path.join(audio_dir, dokid, filename)

        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        else:
            if os.path.exists(dst):
                logging.info(f"File already in destination directory: {dst}")
            else:
                logging.info(f"File not found: {src}")


def audiofile_exists(dokid, filename, audio_dir="data/audio"):
    """
    Check if audio file exists.
    Can exist in either {audio_dir}/{dokid}/{filename} or {audio_dir}/{filename}.

    Args:
        dokid (str): The dokid of the speech. dokid is the directory where the
            audiofile is moved after being download.
        filename (str): The filename of the audio file.
        directory (str): The directory where the audio files are stored.

    Returns:
        bool: True if audio file exists, otherwise False.
    """

    src = os.path.join(audio_dir, filename)
    dst = os.path.join(audio_dir, dokid, filename)

    if os.path.exists(src) or os.path.exists(dst):
        return True
    else:
        return False
