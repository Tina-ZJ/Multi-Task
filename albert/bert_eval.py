#coding:utf-8
import os
import sys
import csv
import codecs
import numpy as np
import time

sys.path.append('../')

import tensorflow as tf
from preprocess import tokenization
from preprocess import bert_data_utils
BASE_DIR=os.path.abspath(os.path.dirname(__file__))

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean("do_lower_case", True, "Whether to lower case the input text")
flags.DEFINE_string("vocab_file", BASE_DIR+"/output"+'/vocab.txt', "vocab file")
flags.DEFINE_string("label_file", BASE_DIR+"/example/bert_checkpoint_cdf.sample/checkpoints"+'/labels.txt', "label file")
flags.DEFINE_string("label_map_file", BASE_DIR+"/output"+'/label_map', "label map file")
flags.DEFINE_string("model_dir", BASE_DIR+"/output/checkpoints", "tf model")
flags.DEFINE_string("bert_config_file", BASE_DIR+"/example/bert_checkpoint_cdf.sample/checkpoints"+'/bert_config.json', "config json file")
tf.flags.DEFINE_string("test_data_file", BASE_DIR+"/predict/test.txt", "Test data source.")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("top", 5, "predict top cid")
tf.flags.DEFINE_integer("max_sequence_length", 30, "max sequnce length")
tf.flags.DEFINE_float("threshold", 0.02, "threshold for predict")
tf.flags.DEFINE_string("cid3_file", '../data_cdf.sample/cid3_name.txt', "cid3 name ")
tf.flags.DEFINE_string("save_file", './predict/predict.txt', "predict file ")

tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def get_cid_name(cid3_file):
    label_name = {}
    with codecs.open(cid3_file, encoding='utf8') as f:
        for line in f:
            line_list = line.strip().split('\t')
            label_name[line_list[1]] = line_list[2]
    return label_name


def parseTag(label_list, word_list):
    words=[]
    result = []
    for (label, word) in zip(label_list, word_list):
        tag = label[2:]
        bies = label[0]
        words.append(word)
        if bies == "s" or bies == "e":
            token = "".join(words)
            result.append(token+'|'+tag)
            words = []
    if len(words) > 0:
        token = "".join(words)
        result.append(token+'|'+tag)
    return result  

def eval():
    f = open(FLAGS.save_file, 'w+')
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    label_map, idx2label = bert_data_utils.read_ner_label_map_file(FLAGS.label_map_file)
    #label_name = get_cid_name(FLAGS.cid3_file)
    batch_datas = bert_data_utils.get_data_yield(FLAGS.test_data_file, 
                                                                                label_map,
                                                                                FLAGS.max_sequence_length,
                                                                                tokenizer,
                                                                                FLAGS.batch_size)

    print('\nEvaluating...\n')

    #Evaluation
    # checkpoint_file = tf.train.latest_checkpoint(FLAGS.model_dir)
    graph = tf.Graph()
    with graph.as_default():
        #restore for tensorflow pb style
        # restore_graph_def = tf.GraphDef()
        # restore_graph_def.ParseFromString(open(FLAGS.model_dir+'/frozen_model.pb', 'rb').read())
        # tf.import_graph_def(restore_graph_def, name='')

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        #restore for tf checkpoint style
        cp_file = tf.train.latest_checkpoint(FLAGS.model_dir)
        # saver = tf.train.Saver()
        saver = tf.train.import_meta_graph('{}.meta'.format(cp_file))
        saver.restore(sess,cp_file)
        
        with sess.as_default():
            #tensors we feed
            input_ids = graph.get_operation_by_name('input_ids').outputs[0]
            input_mask = graph.get_operation_by_name('input_mask').outputs[0]
            token_type_ids = graph.get_operation_by_name('segment_ids').outputs[0]
            #is_training = graph.get_operation_by_name('is_training').outputs[0]
            
            #tensors we want to evaluate
            # precision =  graph.get_operation_by_name('accuracy/precision').outputs[0]
            # recall = graph.get_operation_by_name('accuracy/recall').outputs[0]
            # f1 = graph.get_operation_by_name('accuracy/f1').outputs[0]
            predictions = graph.get_operation_by_name('loss/crf_pred_label_ids').outputs[0]
            cid3_predictions = graph.get_operation_by_name('loss/predictions').outputs[0]


            #collect the predictions here
            t0 = time.time()
            for batch in batch_datas:
                feed_input_ids, feed_input_mask, feed_segment_ids, querys = batch

                feed_dict = {input_ids: feed_input_ids,
                             input_mask: feed_input_mask,
                             token_type_ids: feed_segment_ids,}

                batch_predictions, bath_cid3_predictions = sess.run([predictions,cid3_predictions],feed_dict)
                for  prediction, query in zip(batch_predictions, querys):
                    t =0
                    label_list = []
                    word_list = []
                    for index, token in zip(prediction, query):
                        label = idx2label[index]
                        if label=='[CLS]' or label=='[SEP]':
                            continue
                        label_list.append(label)
                        word_list.append(token)
                    result = parseTag(label_list, word_list)
                    f.write(''.join(query[1:-1])+'\t'+','.join(result)+'\n')
            t1 = time.time()
            
            print(str(t1-t0))

if __name__ == '__main__':
    eval()
