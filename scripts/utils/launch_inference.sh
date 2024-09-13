for i in {0..7}
do
    echo "Running language detection on GPU $i"
    python scripts/lang_detect_whisper.py --num_shards 8 --data_shard $i --gpu_id $i & 
done