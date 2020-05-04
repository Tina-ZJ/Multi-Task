# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
tf.reset_default_graph()
import numpy as np
from data_util import load_test,create_term,create_label, load_cid
from tflearn.data_utils import pad_sequences
import os
import codecs
from model_server import Model

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (
    signature_constants, signature_def_utils, tag_constants, utils)
from tensorflow.python.util import compat
import argparse

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("ckpt","./checkpoint/HAN_300d","checkpoint location for the model")
tf.app.flags.DEFINE_string("pb_path","./model_v1","checkpoint location for the model")
tf.app.flags.DEFINE_integer("num_classes",6059,"number of label")
tf.app.flags.DEFINE_integer("num_count",20,"number of label")
tf.app.flags.DEFINE_integer("num_tags",12,"number of tags")
tf.app.flags.DEFINE_boolean("char_attention_flag",False,"whether to use char attention")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir","checkpoint/HAN_300d/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",8,"max sentence length")
tf.app.flags.DEFINE_integer("chars_length",16,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",False,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")
tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
tf.app.flags.DEFINE_boolean("count_flag",False,"whether to sigle class header.")
tf.app.flags.DEFINE_boolean("tag_flag",True,"whether to tag predict.")
tf.app.flags.DEFINE_integer("num_sentences", 1, "number of sentences in the document")
tf.app.flags.DEFINE_integer("hidden_size",300,"hidden size")
tf.app.flags.DEFINE_string("predict_target_file","data/test_sample_new_JDSeg_checked.txt","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'./HAN_0.1',"target file path for final prediction")
tf.app.flags.DEFINE_float("threshold", 0.01, "test threshold")
tf.app.flags.DEFINE_string("label_index_path","data/cid3_name.txt",'path of cid3')
tf.app.flags.DEFINE_string("term_index_path","data/term_index.txt",'path of term')
tf.app.flags.DEFINE_string("char_index_path","data/char_index.txt",'path of term')
tf.app.flags.DEFINE_integer("loss_number",0,"the loss kinds: 0 for cross entropy loss; 1 for pairwise loss; others for posweight loss which make balance of recall and precision ")

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)

def export_pb_model():
    vocabulary_word2index, vocabulary_index2word= create_term(FLAGS.term_index_path)
    vocabulary_char2index, vocabulary_index2char= create_term(FLAGS.char_index_path)
    vocab_size = len(vocabulary_word2index)
    char_size = len(vocabulary_char2index)
    with tf.Graph().as_default():
        model = Model(FLAGS.loss_number, FLAGS.num_count,FLAGS.num_classes, FLAGS.num_tags, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                                          FLAGS.decay_rate, FLAGS.sequence_length, FLAGS.chars_length,
                                          FLAGS.num_sentences, vocab_size, char_size,FLAGS.embed_size, FLAGS.hidden_size,
                                          FLAGS.is_training, tag_flag=FLAGS.tag_flag, multi_label_flag=FLAGS.multi_label_flag, char_attention_flag=FLAGS.char_attention_flag, count_flag=FLAGS.count_flag)
        model_signature = signature_def_utils.build_signature_def(
            inputs={
                "terms_ids": utils.build_tensor_info(model.input_x),
                "keep_prob_hidden": utils.build_tensor_info(model.dropout_keep_prob)
            },
            outputs={
                "prediction": utils.build_tensor_info(model.predictions),
                "tag_predictions": utils.build_tensor_info(model.tag_predictionss),
                "attention": utils.build_tensor_info(model.attention)
            },
            method_name=signature_constants.CLASSIFY_METHOD_NAME
        )

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )

        sess = tf.Session(config=session_conf)
        saver = tf.train.Saver()
        #saver.restore(sess,_ckpt+'model.ckpt-6')
        print(tf.train.latest_checkpoint(FLAGS.ckpt))
        saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt))
    
        builder = saved_model_builder.SavedModelBuilder(FLAGS.pb_path)
        builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                'cat_han_model_signature':
                    model_signature,
            }
        )
    
        builder.save()
    



if __name__ == "__main__":
    export_pb_model()
