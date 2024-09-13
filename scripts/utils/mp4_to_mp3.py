# Convert all files in /media/fatrek/RDF-18TB/video/RD/A
# from mp4 to mp3
# ffmpeg -i input.mp4 -vn -acodec libmp3lame -q:a 4 output.mp3

import multiprocessing as mp
import os

from tqdm import tqdm

mp4_files = os.listdir("/media/fatrek/RDF-18TB/video/RD/A")


def convert_mp4_to_mp3(file):
    if file.endswith(".mp4"):
        file_name = file.replace(".mp4", ".mp3")
        os.system(
            f"ffmpeg -i /media/fatrek/RDF-18TB/video/RD/A/{file} -vn -acodec libmp3lame -q:a 3 -threads 4 -n data/audio/from_video/{file_name}"
        )
    else:
        print(f"Skipping {file}")
        return


with mp.Pool(5) as pool:
    _ = list(tqdm(pool.imap(convert_mp4_to_mp3, mp4_files), total=len(mp4_files)))
