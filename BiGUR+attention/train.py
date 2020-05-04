# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from model import Model

from data_util import create_term
#from tflearn.data_utils import  pad_sequences
import os
import codecs
import batch_read_tfrecord
import common
import traceback
#import word2vec
#configuration
FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes",6059,"number of label")
tf.app.flags.DEFINE_integer("num_tags",12,"number of tags")
tf.app.flags.DEFINE_integer("num_count",20,"predict max number of label for sigle class header")
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size for training/evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 24000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir","checkpoint/HAN_300d/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sequence_length",8,"max sentence length")
tf.app.flags.DEFINE_integer("loss_number",0,"the loss kinds: 0 for cross entropy loss; 1 for pairwise loss; others for posweight loss which make balance of recall and precision ")
tf.app.flags.DEFINE_integer("chars_length",16,"max chars length")
tf.app.flags.DEFINE_integer("char_size",10000,"how many of chars")
tf.app.flags.DEFINE_integer("embed_size",300,"embedding size")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_integer("validate_step", 1000, "how many step to validate.") 
tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("train_sample_file","data/train_sample.tfrecord","path of traning data.")
tf.app.flags.DEFINE_string("dev_sample_file","data/dev_sample.tfrecord","path of dev data.")
tf.app.flags.DEFINE_string("word2vec_model_path","data/term_embedding.txt","term embeddings")
tf.app.flags.DEFINE_boolean("multi_label_flag",True,"use multi label or single label.")
tf.app.flags.DEFINE_boolean("char_attention_flag",False,"whether to use char attention")
tf.app.flags.DEFINE_boolean("count_flag",False,"whether to sigle class header")
tf.app.flags.DEFINE_boolean("tag_flag",True,"whether to sigle class header")
tf.app.flags.DEFINE_integer("num_sentences", 1, "number of sentences in the document")
tf.app.flags.DEFINE_integer("hidden_size",300,"hidden size")
tf.app.flags.DEFINE_string("term_index_path","data/term_index.txt",'path of term')
tf.app.flags.DEFINE_string("char_index_path","data/char_index.txt",'path of term')
tf.app.flags.DEFINE_string("label_index_path","data/cid3_name.txt",'path of term')
tf.app.flags.DEFINE_string("sample_size_file","data/sample_size.txt",'path of term')
tf.app.flags.DEFINE_string("summary_dir","./summary/",'path of summary')
tf.app.flags.DEFINE_integer("train_sample_num",65657476,"train sample num")
tf.app.flags.DEFINE_integer("dev_sample_num",1339949,"dev sample num")

def main(_):
    #1. load vocabulary
    vocabulary_word2index, vocabulary_index2word= create_term(FLAGS.term_index_path)
    vocab_size = len(vocabulary_index2word)
    char_size = FLAGS.char_size
    print("vocab_size:",vocab_size)
    if FLAGS.char_attention_flag:
        vocabulary_char2index, vocabulary_index2char= create_term(FLAGS.char_index_path)
        char_size = len(vocabulary_index2char)
        print("char_size:",char_size)
    #2.create session.
    with tf.Graph().as_default():
        config=tf.ConfigProto()
        config.gpu_options.allow_growth=True
        model = Model( FLAGS.loss_number, FLAGS.num_count, FLAGS.num_classes, FLAGS.num_tags, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                                      FLAGS.decay_rate, FLAGS.sequence_length, FLAGS.chars_length,
                                      FLAGS.num_sentences, vocab_size, char_size, FLAGS.embed_size, FLAGS.hidden_size,
                                      FLAGS.is_training, tag_flag=FLAGS.tag_flag, multi_label_flag=FLAGS.multi_label_flag, char_attention_flag=FLAGS.char_attention_flag, count_flag=FLAGS.count_flag)
        train_batcher = batch_read_tfrecord.SegBatcher(FLAGS.train_sample_file, FLAGS.batch_size,num_epochs=FLAGS.num_epochs)
        dev_batcher = batch_read_tfrecord.SegBatcher(FLAGS.dev_sample_file, FLAGS.batch_size,num_epochs=FLAGS.num_epochs)
        global_init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            #tensorboard
            train_writer = tf.summary.FileWriter(FLAGS.summary_dir + 'train/', sess.graph)
            dev_writer = tf.summary.FileWriter(FLAGS.summary_dir + 'dev/')
            sess.run(global_init_op)
            sess.run(local_init_op)
            if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
                print("Restoring Variables from Checkpoint")
                saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            else:
                print('Initializing Variables')
                if FLAGS.use_embedding:
                    assign_pretrained_word_embedding(sess, vocabulary_index2word, vocab_size, model,word2vec_model_path=FLAGS.word2vec_model_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            curr_epoch=sess.run(model.epoch_step)
            #3.feed data & training
            train_sample_num = FLAGS.train_sample_num
            dev_sample_num = FLAGS.dev_sample_num
            best_eval_f1 = 0

            for epoch in range(curr_epoch,FLAGS.num_epochs):
                loss, acc, tag_acc, recall, precision, f1, counter = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
                eval_loss, eval_acc, eval_tag_acc, eval_recall, eval_precision, eval_f1, counter_dev = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
                train_example_num = 0
                dev_example_num = 0
                while train_example_num < train_sample_num:
                    try :
                        train_batch_data = sess.run(train_batcher.next_batch_op)
                        trainX, trainXChar, trainY, trainTags = train_batch_data
                        if len(trainX)!=FLAGS.batch_size:
                            continue
                        #trainX = pad_sequences(trainX, maxlen=FLAGS.sequence_length, value=0.)
                        trainY = common.get_one_hot_label(trainY, FLAGS.num_classes)
                        if FLAGS.char_attention_flag:
                            #trainXChar = pad_sequences(trainXChar, maxlen=FLAGS.chars_length, value=0.)
                            feed_dict = {model.input_x: trainX, model.input_char: trainXChar, model.dropout_keep_prob: 0.5}
                        else:
                            feed_dict = {model.input_x: trainX, model.dropout_keep_prob: 0.5}

                        if not FLAGS.multi_label_flag:
                            feed_dict[model.input_y] = trainY
                        else:
                            feed_dict[model.input_y_multilabel]=trainY
                
                        # for tag head
                        if FLAGS.tag_flag:
                            feed_dict[model.input_tag] = trainTags
                        

                        summary_merge, global_step, curr_loss,curr_acc, curr_tag_acc, curr_f1, curr_recall, curr_precision, _=sess.run([model.summary_merge, model.global_step, model.loss_val,model.accuracy,model.tag_acc, model.f1,model.recall,model.precision,model.train_op],feed_dict)

                        train_writer.add_summary(summary_merge, global_step)

                        loss,counter,acc, tag_acc, recall, precision, f1 =loss+curr_loss,counter+1,acc+curr_acc, tag_acc+curr_tag_acc,recall+curr_recall, precision+curr_precision, f1+curr_f1
                        train_example_num += FLAGS.batch_size
                        if counter %2000==0:
                            print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f\tTrain Tag Acc:%.3f\tTrain precision:%.3f\tTrain recall:%.3f\tTrain f1:%.3f" %(epoch,counter,loss/float(counter),acc/float(counter), tag_acc/float(counter), precision/float(counter),recall/float(counter),f1/float(counter)))
                    except tf.errors.OutOfRangeError:
                        print("Done Training")
                        break

                    ##VALIDATION VALIDATION VALIDATION PART######################################################################################################
                while dev_example_num < dev_sample_num:
                    counter_dev+=1
                    try:
                        dev_batch_data = sess.run(dev_batcher.next_batch_op)
                        testX, testXChar, testY, testTag = dev_batch_data
                        if len(testX)!=FLAGS.batch_size:
                            continue
                        #testX = pad_sequences(testX, maxlen=FLAGS.sequence_length, value=0.)
                        testY = common.get_one_hot_label(testY, FLAGS.num_classes)
                        if FLAGS.char_attention_flag:
                            testXChar = pad_sequences(testXChar, maxlen=FLAGS.chars_length, value=0.)
                        else:
                            testXChar=[]
                        cur_eval_loss, cur_eval_acc, cur_eval_tag_acc, cur_eval_precision, cur_eval_recall, cur_eval_f1 = do_eval(dev_writer, sess, model, testX, testXChar, testY, testTag)
                        eval_loss,eval_acc, eval_tag_acc, eval_recall, eval_precision, eval_f1 = eval_loss+cur_eval_loss,eval_acc+cur_eval_acc, eval_tag_acc+cur_eval_tag_acc, eval_recall+cur_eval_recall, eval_precision+cur_eval_precision, eval_f1+cur_eval_f1
                        dev_example_num += FLAGS.batch_size
                        if counter_dev % FLAGS.validate_step == 0:
                            print("Epoch %d \tBatch %d\tValidation Loss:%.3f\tValidation Accuracy: %.3f\tValidation Tag Acc: %.3f\tValidation precision:%.3f\tValidation recall:%.3f\tValidation f1:%.3f\t" % (epoch,counter_dev,eval_loss/float(counter_dev),eval_acc/float(counter_dev),eval_tag_acc/float(counter_dev), eval_precision/float(counter_dev),eval_recall/float(counter_dev),eval_f1/float(counter_dev)))
                            if eval_f1 > best_eval_f1:
                                best_eval_f1=eval_f1
                                print ("Validation best f1: %f" % best_eval_f1)
                    except tf.errors.OutOfRangeError:
                        print("Done test")
                        break

            ##VALIDATION VALIDATION VALIDATION PART######################################################################################################

                #epoch increment
                print("going to increment epoch counter....")
                sess.run(model.epoch_increment)

                # 4.validation
                print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
                if epoch % FLAGS.validate_every==0:
                    #save model to checkpoint
                    if not os.path.exists(FLAGS.ckpt_dir):
                        os.makedirs(FLAGS.ckpt_dir)
                    save_path=FLAGS.ckpt_dir+"model.ckpt"
                    saver.save(sess,save_path,global_step=epoch)
        coord.request_stop()
        coord.join(threads)
        sess.close()

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,model,word2vec_model_path=None):
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    word2vec_model= word2vec.load(word2vec_model_path)
    # word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(model.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

# validate 
def do_eval(dev_writer, sess,model,evalX,evalXChar,evalY, testTag):
    if FLAGS.char_attention_flag:
        feed_dict = {model.input_x: evalX, model.input_char: evalXChar,  model.dropout_keep_prob: 1}
    else:
        feed_dict = {model.input_x: evalX, model.dropout_keep_prob: 1}
        
    if not FLAGS.multi_label_flag:
        feed_dict[model.input_y] = evalY
    else:
        feed_dict[model.input_y_multilabel] = evalY
        feed_dict[model.input_tag] = testTag
    dev_summary_merge, global_step, curr_eval_precision, curr_eval_recall, curr_eval_f1, curr_eval_loss, logits,curr_eval_acc, curr_eval_tag_acc= sess.run([model.summary_merge, model.global_step, model.precision, model.recall, model.f1, model.loss_val,model.logits,model.accuracy, model.tag_acc],feed_dict)
    dev_writer.add_summary(dev_summary_merge, global_step)
    return curr_eval_loss, curr_eval_acc, curr_eval_tag_acc, curr_eval_precision, curr_eval_recall, curr_eval_f1


if __name__ == "__main__":
    tf.app.run()
