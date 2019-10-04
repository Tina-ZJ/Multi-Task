# -*- coding: utf-8 -*-
import sys
import tensorflow as tf
import time
tf.reset_default_graph()
import numpy as np
from data_util import load_test,create_term,create_label, load_cid, load_tag
from tflearn.data_utils import pad_sequences
import os
import codecs
import time
from model_server import Model

#configuration
FLAGS=tf.app.flags.FLAGS
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
tf.app.flags.DEFINE_integer("num_epochs",7,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_boolean("use_embedding",True,"whether to use embedding or not.")
tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
tf.app.flags.DEFINE_boolean("count_flag",False,"whether to sigle class header.")
tf.app.flags.DEFINE_boolean("tag_flag",True,"whether to tag predict.")
tf.app.flags.DEFINE_integer("num_sentences", 1, "number of sentences in the document")
tf.app.flags.DEFINE_integer("hidden_size",300,"hidden size")
tf.app.flags.DEFINE_string("predict_target_file","./test.txt","target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file",'./test.predict',"target file path for final prediction")
tf.app.flags.DEFINE_float("threshold", 0.5, "test threshold")
tf.app.flags.DEFINE_string("label_index_path","data/cid3_name.txt",'path of cid3')
tf.app.flags.DEFINE_string("tags_index_path","data/tags_index.txt",'path of tags')
tf.app.flags.DEFINE_string("term_index_path","data/term_index.txt",'path of term')
tf.app.flags.DEFINE_string("char_index_path","data/char_index.txt",'path of term')
tf.app.flags.DEFINE_integer("loss_number",0,"the loss kinds: 0 for cross entropy loss; 1 for pairwise loss; others for posweight loss which make balance of recall and precision ")

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)

def main(_):
    # 1.load data with vocabulary of words and labels
    vocabulary_word2index, vocabulary_index2word= create_term(FLAGS.term_index_path)
    vocabulary_char2index, vocabulary_index2char= create_term(FLAGS.char_index_path)
    # for tags index
    tags_index2word = load_tag(FLAGS.tags_index_path)
    vocab_size = len(vocabulary_word2index)
    char_size = len(vocabulary_char2index)
    vocabulary_word2index_label, vocabulary_index2word_label = create_label(FLAGS.label_index_path)
    FLAGS.num_classes = len(vocabulary_word2index_label)
    FLAGS.num_tags = len(tags_index2word)
    testX, testXChar, lines = load_test(FLAGS.predict_target_file, vocabulary_word2index, vocabulary_char2index)

    # 2.Data preprocessing: Sequence padding
    print("start padding....")
    #testX = pad_sequences(testX, maxlen=FLAGS.sequence_length, value=0.)  # padding to max length
    if FLAGS.char_attention_flag:
        testXChar = pad_sequences(testXChar, maxlen=FLAGS.chars_length, value=0.)
    print("end padding...")
   # 3.create session.
    with tf.Graph().as_default():
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            # 4.Instantiate Model
            model = Model(FLAGS.loss_number, FLAGS.num_count,FLAGS.num_classes, FLAGS.num_tags, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                                          FLAGS.decay_rate, FLAGS.sequence_length, FLAGS.chars_length,
                                          FLAGS.num_sentences, vocab_size, char_size,FLAGS.embed_size, FLAGS.hidden_size,
                                          FLAGS.is_training, tag_flag=FLAGS.tag_flag, multi_label_flag=FLAGS.multi_label_flag, char_attention_flag=FLAGS.char_attention_flag, count_flag=FLAGS.count_flag)
            saver=tf.train.Saver()
            if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
                print("Restoring Variables from Checkpoint")
                saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
                #saver.restore(sess,FLAGS.ckpt_dir+'model.ckpt-2')
            else:
                print("Can't find the checkpoint.going to stop")
                return
            # 5.feed data, to get logits
            number_of_training_data=len(testX);print("number_of_training_data:",number_of_training_data)
            predict_target_file_f = codecs.open(FLAGS.predict_source_file, 'a', 'utf8')
            label_name = load_cid(FLAGS.label_index_path)
            t0 = time.time()
            for start, end in zip(range(0, number_of_training_data, FLAGS.batch_size),range(FLAGS.batch_size, number_of_training_data+1, FLAGS.batch_size)):
                if FLAGS.char_attention_flag:
                    predictions=sess.run(model.predictions,feed_dict={model.input_x:testX[start:end],model.input_char:testXChar[start:end],model.dropout_keep_prob:1})
                else:
                    attention, predictions, tag_predictions=sess.run([model.attention,model.predictions,model.tag_predictions],feed_dict={model.input_x:testX[start:end],model.dropout_keep_prob:1})
                lines_sublist=lines[start:end]
                get_label_using_logits_batch(lines_sublist, attention, predictions, tag_predictions, tags_index2word, vocabulary_index2word_label, predict_target_file_f,FLAGS.threshold, label_name)
            t1 = time.time()
            print ("all running time: %s " % str(t1-t0))
            predict_target_file_f.close()


# get label using logits
def get_label_using_logits_batch(lines_sublist,attention, predictions_batch, tag_predictions, tags_index2word, vocabulary_index2word_label,f,threshold, label_name):
    for i,predictions in enumerate(predictions_batch):
        predictions_sorted = sorted(predictions, reverse=True)
        index_sorted = np.argsort(-predictions)
        label_list = []
        label_names = []
        tags_label = []
        #for tag
        tag_prediction = tag_predictions[i]
        for tag in tag_prediction:
            tags_label.append(tags_index2word[str(tag)])
        for index, predict in zip(index_sorted, predictions_sorted):
            if predict > threshold:
                label = vocabulary_index2word_label[index]
                label_score = label+':'+str(predict)
                label_list.append(label_score)
                label_names.append(label_name[label])
            if len(label_list)==0:
                label_list.append('0:0.0')
                label_names.append('填充类')
        write_question_id_with_labels(lines_sublist[i], attention[i], label_list, tags_label, f, label_names)
    f.flush()

def write_question_id_with_labels(question_id,attention, labels_list,tags_label, f, label_names):
    labels_string=",".join(labels_list)
    tags_string=",".join(tags_label)
    label_string_name = ','.join(label_names)
    attention_list = ','.join(map(str, attention))
    f.write(question_id+"\t"+tags_string+'\t'+labels_string+ '\t' + label_string_name + "\t"+attention_list+"\n")

if __name__ == "__main__":
    tf.app.run()
