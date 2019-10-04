#!/bin/bash
startTime=`date +"%Y-%m-%d %H:%M:%S"`
start_seconds=$(date +%s)

# data process
python preprocess/term.py data/all_sample.txt data/term_index.txt data/cid3_name.txt data/char_index.txt data/tags_index.txt

# split corpus to train and dev
python split_sample.py
if [ $? -ne 0 ]
then
    echo "split corpus to train and dev failed "
    exit -1
else
    echo "split corpus to train and dev Done "
fi

# transfer data to tfrecord format
python -u transfer_sample_tfrecord.py
if [ $? -ne 0 ]
then
    echo "transfer data to tfrecord failed"
    exit -1
else
    echo "transfer data to tfrecord Done "
fi

# begain train model
num_classes=`awk -F'\t' 'END{print $1}' data/cid3_name.txt`
num_tags=`awk -F'\t' 'END{print $2+1}' data/tags_index.txt`

python -u train.py --num_classes=${num_classes} --num_tags=${num_tags} --embed_size=300 -- --decay_steps=24000 --validate_every=1 --num_epochs=8
if [ $? -ne 0 ]
then
    echo " train HAN model failed"
    exit -1
else
    echo "train HAN model Done"
fi

endTime=`date +"%Y-%m-%d %H:%M:%S"`

# excute time
end_seconds=$(date +%s)
useSeconds=$[$end_seconds - $start_seconds]
useHours=$[$useSeconds / 3600]

echo " the script running time: $startTime ---> $endTime : $useHours hours "

