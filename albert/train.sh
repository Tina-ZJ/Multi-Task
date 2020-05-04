startTime=`date +"%Y-%m-%d %H:%M:%S"`
start_seconds=$(date +%s)
export MODEL_DIR=./albert_tiny_zh

python run_sequencelabeling.py --task_name=ner \
--output_dir=./output \
--data_dir=./data \
--init_checkpoint=$MODEL_DIR/albert_model.ckpt \
--bert_config_file=$MODEL_DIR/albert_config_tiny.json \
--vocab_file=$MODEL_DIR/vocab.txt \
--max_seq_length=64  \
--num_train_epochs=5 \
--learning_rate=1e-4  \
--train_batch_size=64 \
--do_predict=False \
--do_eval=False \
--do_train=True \

endTime=`date +"%Y-%m-%d %H:%M:%S"`
end_seconds=$(date +%s)
useSeconds=$[$end_seconds - $start_seconds]
useHours=$[$useSeconds / 3600]

echo " the script running time: $startTime ---> $endTime : $useHours hours " 
