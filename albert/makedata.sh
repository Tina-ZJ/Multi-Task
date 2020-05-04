#coding:utf-8
startTime=`date +"%Y-%m-%d %H:%M:%S"`
start_seconds=$(date +%s)

export MODEL_DIR=./albert_tiny_zh

python create_pretraining_data.py  \
                         --input_file=./pretrain_data/sample.txt \
                         --output_file=./tmp/train.tfrecord \
                         --vocab_file=$MODEL_DIR/vocab.txt \
                         --do_lower_case=True \
                         --max_seq_length=50  \
                         --max_predictions_per_seq=10 \
                         --masked_lm_prob=0.20 \
                         --random_seed=12345 \
                         --dupe_factor=5

endTime=`date +"%Y-%m-%d %H:%M:%S"`
end_seconds=$(date +%s)
useSeconds=$[$end_seconds - $start_seconds]
useHours=$[$useSeconds / 3600]
echo " the script running time: $startTime ---> $endTime : $useHours hours "
