startTime=`date +"%Y-%m-%d %H:%M:%S"`
start_seconds=$(date +%s)
input=$1
output=$2

python bert_eval_pb.py --test_data_file=${input} --save_file=${output}

python predict/cdf.py ${output} ${output}.cdf

endTime=`date +"%Y-%m-%d %H:%M:%S"`
end_seconds=$(date +%s)
useSeconds=$[$end_seconds - $start_seconds]
useHours=$[$useSeconds / 3600]

echo " the script running time: $startTime ---> $endTime : $useHours hours " 
