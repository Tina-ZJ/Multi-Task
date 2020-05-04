#coding:utf-8
import os
import sys
import csv
import codecs
import numpy as np

sys.path.append('../')

import time
import tensorflow as tf
from preprocess import tokenization
from preprocess import bert_data_utils
from tensorflow.contrib import predictor
from pathlib import Path

BASE_DIR=os.path.abspath(os.path.dirname(__file__))
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean("do_lower_case", True, "Whether to lower case the input text")
flags.DEFINE_string("vocab_file", BASE_DIR+"/output"+'/vocab.txt', "vocab file")
flags.DEFINE_string("label_file", BASE_DIR+"/example/bert_checkpoint_cdf.sample/checkpoints"+'/labels.txt', "label file")
flags.DEFINE_string("label_map_file", BASE_DIR+"/output"+'/label_map', "label map file")
flags.DEFINE_string("label_cid3_map_file", BASE_DIR+"/output"+'/label_cid3_map', "label map file")
flags.DEFINE_string("model_dir", BASE_DIR+"/output/checkpoints", "tf model")
flags.DEFINE_string("bert_config_file", BASE_DIR+"/example/bert_checkpoint_cdf.sample/checkpoints"+'/bert_config.json', "config json file")
tf.flags.DEFINE_string("test_data_file", BASE_DIR+"/predict/test.txt", "Test data source.")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("top", 5, "predict top cid")
tf.flags.DEFINE_integer("max_sequence_length", 110, "max sequnce length")
tf.flags.DEFINE_float("threshold", 0.02, "threshold for predict")
tf.flags.DEFINE_string("cid3_file", './data/labels_cid3.tsv', "cid3 name ")
tf.flags.DEFINE_string("save_file", './predict/predict.txt', "predict file ")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def get_cid_name(cid3_file):
    label_name = {}
    with codecs.open(cid3_file, encoding='utf8') as f:
        for line in f:
            line_list = line.strip().split('\t')
            label_name[str(line_list[1])] = line_list[2]
    return label_name


def parseTag(label_list, word_list, weight_list):
    words=[]
    result = []
    result2 = []
    term_weight = 0.0
    tags = dict()
    for (label, word, weight) in zip(label_list, word_list, weight_list):
        tag = label[2:]
        bies = label[0]
        words.append(word)
        term_weight+=float(weight)
        if bies == "s" or bies == "e":
            token = "".join(words)
            result.append(token+':'+tag)
            result2.append(token+':'+str(term_weight))
            words = []
            term_weight = 0.0
    if len(words) > 0:
        token = "".join(words)
        result.append(token+':'+tag)
        result2.append(token+':'+str(term_weight))
    return result, result2  

def eval():
    f = open(FLAGS.save_file, 'w+')
    # path
    subdirs = [x for x in Path(FLAGS.model_dir).iterdir() if x.is_dir() and 'temp' not in str(x)]
    model_pb = str(sorted(subdirs)[-1])
 
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    label_map, idx2label = bert_data_utils.read_ner_label_map_file(FLAGS.label_map_file)
    label_cid3_map, idx2label_cid3 = bert_data_utils.read_ner_label_map_file(FLAGS.label_cid3_map_file)
    label_name = get_cid_name(FLAGS.cid3_file)
    batch_datas = bert_data_utils.get_data_yield(FLAGS.test_data_file, 
                                                                                label_map,
                                                                                FLAGS.max_sequence_length,
                                                                                tokenizer,
                                                                                FLAGS.batch_size)

    predict_fn = predictor.from_saved_model(model_pb)
    t0 = time.time()
    for batch in batch_datas:
        feed_input_ids, feed_input_mask, feed_segment_ids, querys = batch

        feed_dict = {'input_ids': feed_input_ids,
                     'input_mask': feed_input_mask,
                     'segment_ids': feed_segment_ids}
        result = predict_fn(feed_dict)
        batch_predictions = result['pred_label_ids']
        batch_char_weight = result['char_weight']
        batch_predictions_cid3 = result['predictions']
        #continue 
        for  prediction, query, predictions_cid3, char_weights in zip(batch_predictions, querys, batch_predictions_cid3, batch_char_weight):
            predictions_sorted = sorted(predictions_cid3, reverse=True)
            index_sorted = np.argsort(-predictions_cid3)
            label_list = []
            label_scores = []
            label_names = []
            word_list = []
            weight_list = []
            for index, predict in zip(index_sorted, predictions_sorted):
                if predict > FLAGS.threshold:
                    label = idx2label_cid3[index] 
                    name = label_name[label]
                    label_names.append(name)
                    label_scores.append(label+':'+str(predict))
                    
                if len(label_scores)==0:
                    label_scores.append('0'+':'+str(0.0))
                    label_names.append('填充类')
            for index, token, char_weight in zip(prediction, query, char_weights):
                label = idx2label[index]
                if label=='[CLS]' or label=='[SEP]':
                    continue
                label_list.append(label)
                word_list.append(token)
                #weight_list.append(token+':'+str(char_weight))
                weight_list.append(char_weight)
            result, result2 = parseTag(label_list, word_list, weight_list)

        #f.write(''.join(query[1:-1])+'\t'+' '.join(label_list)+'\t'+','.join(result2)+'\t'+','.join(result)+'\t'+','.join(label_scores)+'\t'+','.join(label_names)+'\n')
        f.write(''.join(query[1:-1])+'\t'+','.join(result)+'\t'+','.join(label_scores)+'\t'+','.join(label_names)+'\t'+','.join(result2)+'\n')

    t1 = time.time()
    print(str(t1-t0))
if __name__ == '__main__':
    eval()
