#!/bin/bash

for i in {0..7}
do
    echo "Running language detection on GPU $i"
    python scripts/transcribe_wav2vec2.py --num_shards 8 --data_shard $i --gpu_id $i \
            --json_dir="/shared/delat/audio/riksdagen/data/vad_output_web" \
            --output_dir="/workspace/rixvox/data/vad_wav2vec_output" &
done

# Wait for all background jobs to finish
wait