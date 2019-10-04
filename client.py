# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
tf.reset_default_graph()
import numpy as np
from data_util import load_test,create_term,create_label, load_cid
from tflearn.data_utils import pad_sequences
import os
import codecs
from HAN_model import HierarchicalAttention

#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",5893,"number of label")
tf.app.flags.DEFINE_integer("num_count",20,"number of label")
tf.app.flags.DEFINE_integer("top_number",10,"predict top k label")
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
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_boolean("count_flag",False,"whether to sigele class header.")
tf.app.flags.DEFINE_string("traning_data_path","train-zhihu4-only-title-all.txt","path of traning data.")
tf.app.flags.DEFINE_string("word2vec_model_path","zhihu-word2vec-title-desc.bin-100","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
tf.app.flags.DEFINE_integer("num_sentences", 1, "number of sentences in the document")
tf.app.flags.DEFINE_integer("hidden_size",300,"hidden size")
tf.app.flags.DEFINE_string("predict_target_file","data/validation_3000","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'./validation_result.txt',"target file path for final prediction")
tf.app.flags.DEFINE_float("threshold", 0.3, "test threshold")
tf.app.flags.DEFINE_string("label_index_path","data/cid3_name.txt",'path of cid3')
tf.app.flags.DEFINE_string("term_index_path","data/term_index.txt",'path of term')
tf.app.flags.DEFINE_string("char_index_path","data/char_index.txt",'path of term')
tf.app.flags.DEFINE_integer("loss_number",0,"the loss kinds: 0 for cross entropy loss; 1 for pairwise loss; others for posweight loss which make balance of recall and precision ")


class cid3Client():
    def __init__(self):
        self.vocabulary_word2index, self.vocabulary_index2word= create_term(FLAGS.term_index_path)
        self.vocab_size = len(self.vocabulary_word2index)
        self.char_size = 100
        self.vocabulary_word2index_label, self.vocabulary_index2word_label = create_label(FLAGS.label_index_path)
        self.label_name = load_cid(FLAGS.label_index_path)
        with tf.Graph().as_default():
            config=tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.Session(config=config)
            # 4.Instantiate Model
            self.model = HierarchicalAttention(FLAGS.loss_number, FLAGS.num_count,FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                                          FLAGS.decay_rate, FLAGS.sequence_length, FLAGS.chars_length,
                                          FLAGS.num_sentences, self.vocab_size, self.char_size, FLAGS.embed_size, FLAGS.hidden_size,
                                          FLAGS.is_training, multi_label_flag=FLAGS.multi_label_flag, char_attention_flag=FLAGS.char_attention_flag, count_flag=FLAGS.count_flag)
            self.saver=tf.train.Saver()
            if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
                print("Restoring Variables from Checkpoint")
                self.saver.restore(self.sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
                #self.saver.restore(self.sess,FLAGS.ckpt_dir+'model.ckpt-8')
            else:
                print("Can't find the checkpoint.going to stop")

    def predict(self, input_x):
        predictions_batch,attentions_batch=self.sess.run([self.model.predictions,self.model.attention],feed_dict={self.model.input_x:input_x,self.model.dropout_keep_prob:1})
        predictions = predictions_batch[0]
        attention = attentions_batch[0] 
        predictions_sorted = sorted(predictions, reverse=True)
        predictions_sorted_top50 = predictions_sorted[:FLAGS.top_number]
        index_sorted = np.argsort(-predictions)
        index_sorted_top50 = index_sorted[:FLAGS.top_number]
        label_list = []
        label_names = []
        for index, predict in zip(index_sorted_top50, predictions_sorted_top50):
            label = self.vocabulary_index2word_label[index]
            label_score = label+':'+str(predict)
            label_list.append(label_score)
            label_names.append(self.label_name[label])
        result = ",".join(label_list)+'\t'+",".join(label_names)+'\t'+','.join(map(str,attention))
        return result    
        
    def test(self, line):
        terms = line.strip().split()
        terms_index = [[self.vocabulary_word2index.get(e,1) for e in terms]]
        #input_x = pad_sequences(terms_index,maxlen=FLAGS.sequence_length, value=0.)
        input_x = terms_index
        result = self.predict(input_x)
        return result

 
if __name__ == "__main__":
    client = cid3Client()
    line = input('please input a query:')
    while line !='q':
        if line == '':
            line = input('please input the keyword')
            continue
        print("query:" + line)
        cid3 = client.test(line)
        print("cid3 result:" + cid3)
        line = input('please input the query:')
    print("program exit")

