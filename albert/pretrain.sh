#coding:utf-8
export MODEL_DIR=./albert_tiny_zh

python run_pretraining.py  \
                         --input_file=./tmp/train.tfrecord \
                         --output_dir=./tmp/pretraining_output \
                         --do_train=True \
                         --do_eval=True \
                         --bert_config_file=$MODEL_DIR/albert_config_tiny.json \
                         --train_batch_size=32 \
                         --max_seq_length=50 \
                         --max_predictions_per_seq=10 \
                         --num_train_steps=20 \
                         --num_warmup_steps=10 \
                         --learning_rate=2e-5


