# Covert an mp3 file to a wav file 16khz mono
# Usage: ./mp3_to_wav.sh <input_mp3_file> <output_wav_file>
# Dependencies: ffmpeg

ffmpeg -i $1 -acodec pcm_s16le -ac 1 -ar 16000 $2