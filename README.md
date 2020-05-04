# Ner and Classification for Multi Task

####################### BiGRU+attention ###########################
Data format
1. query \t tags \t classification labels
2. more details please see data/all_sample.txt samples

Train
1. cd BiGRU+attention
1. bash train.sh

Predict
1. python predict.py




######################## albert ################################

Data format
1. query \t tags \t classification labels
2. more details please see data/train.tsv

Prepare
1. Download the pre-trained chinese model from https://github.com/brightmart/albert_zh
2. Or you can pretrain your model use your domain data

Pretrain
followe theses steps:
1. put your pretrain data in pretrain_data directory eg: sample.txt
one sentence a line and documents are separated by an empty line

2. converte data to tfrecord
sh makedata.sh

3. then pretrain your model 
sh pretrain.sh
model are saved in tmp/pretraining_output directory
then put the pretrained model in albert_tiny_zh directory

if you have any doubt, please see more details about pretrain: https://github.com/google-research/bert

Train
1. sh train.sh

Export bp model and predict
1. python export_sign.py
the pb model is saved in output/checkpoints 
2. sh predict.sh $inputfile $outputfile   eg: sh predict.sh test.txt test.albert


