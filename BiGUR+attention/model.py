# -*- coding: utf-8 -*-
import tensorflow as tf
import time
import numpy as np
import tensorflow.contrib as tf_contrib
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import rnn

class Model:
    def __init__(self, loss_number, num_count, num_classes, num_tags, learning_rate, batch_size, decay_steps, decay_rate, sequence_length, char_length, num_sentences,
                 vocab_size, char_size, embed_size,hidden_size, is_training, tag_flag=True, multi_label_flag=True,char_attention_flag=True, count_flag=False,initializer=initializers.xavier_initializer(),clip_gradients=5.0):
        """init all hyperparameter here"""
        # set hyperparamter
        self.loss_number = loss_number
        self.num_count = num_count
        self.num_classes = num_classes
        self.num_tags = num_tags
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.char_length = char_length
        self.num_sentences = num_sentences
        self.vocab_size = vocab_size
        self.char_size = char_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.95)
        self.initializer = initializer
        self.multi_label_flag = multi_label_flag
        self.char_attention_flag = char_attention_flag
        self.count_flag = count_flag
        self.tag_flag = tag_flag
        self.hidden_size = hidden_size
        self.clip_gradients=clip_gradients

        # add placeholder (X,label)
        #self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")
        # sequence true length
        self.length = tf.reduce_sum(tf.sign(self.input_x), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)

        self.input_mask = tf.cast(tf.not_equal(self.input_x, 0), tf.float32)
        if self.char_attention_flag: 
            #self.input_char = tf.placeholder(tf.int32, [None, self.char_length], name="input_char")
            self.input_char = tf.placeholder(tf.int32, [None, None], name="input_char")
            self.input_mask_char = tf.cast(tf.not_equal(self.input_char, 0), tf.float32)

        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")
        self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.num_classes],name="input_y_multilabel")
        # for tag head
        self.input_tag = tf.placeholder(tf.int32, [None, None], name="input_tag")
        
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        self.instantiate_weights()
        # for tag head
        if self.tag_flag:
            self.logits, self.tag_logits = self.inference_simple()
        else:
            self.logits = self.inference_simple()
        self.predictions = tf.reshape(tf.sigmoid(self.logits),[-1, self.num_classes], name='prediction')
        # for tag head
        tag_predictions = tf.nn.softmax(self.tag_logits, axis=-1) 
        tag_predictionss = tf.nn.softmax(self.tag_logitss, axis=-1) 
        self.tag_predictions = tf.argmax(tag_predictions, axis=-1) 
        self.tag_predictionss = tf.argmax(tag_predictionss, axis=-1) 


        if not self.multi_label_flag:
            correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32),self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
        else:
            self.tp = tf.reduce_sum(tf.cast(tf.greater(self.predictions, 0.3), tf.float32) * tf.cast(tf.equal(self.input_y_multilabel, 1), tf.float32))
            self.tn = tf.reduce_sum(tf.cast(tf.less(self.predictions, 0.3), tf.float32) * tf.cast(tf.equal(self.input_y_multilabel, 0), tf.float32))
            self.fp = tf.reduce_sum(tf.cast(tf.greater(self.predictions, 0.3), tf.float32) * tf.cast(tf.equal(self.input_y_multilabel, 0), tf.float32))
            self.fn = tf.reduce_sum(tf.cast(tf.less(self.predictions, 0.3), tf.float32) * tf.cast(tf.equal(self.input_y_multilabel, 1), tf.float32))
            self.accuracy = tf.div(self.tp+self.tn, self.tp+self.tn+self.fp+self.fn, name='accuracy')
            self.precision = tf.div(self.tp, self.tp+self.fp, name='precision')
            self.recall = tf.div(self.tp, self.tp+self.fn, name='recall')
            self.f1 = tf.div(2*self.precision*self.recall, self.precision+self.recall, name='F1')
            # for tag acc
            correct = tf.cast(tf.equal(tf.cast(self.tag_predictions, tf.int32) - self.input_tag, 0), tf.float32)
            correct = tf.reduce_sum(self.input_mask * correct)
            self.tag_acc = tf.div(correct, tf.cast(tf.reduce_sum(self.length), tf.float32))
            
            
        if multi_label_flag:
            print("going to use multi label loss.")
            self.loss_val = self.loss_multilabel()
        else:
            print("going to use single label loss.")
            self.loss_val = self.loss()

        if not is_training:
            return
        if self.count_flag:
            self.predictions_count, self.lcnt_loss, self.acc = self.loss_count()
            self.count_train_op = self.train_count()
        else:
            self.train_op = self.train()
        
        # tensorboard
        tf.summary.scalar("loss", self.loss_val)
        tf.summary.scalar("precision", self.precision)
        tf.summary.scalar("recall", self.recall)
        tf.summary.scalar("F1", self.f1)
        tf.summary.scalar("larning_rate", self.decay_learning_rate)
        self.summary_merge = tf.summary.merge_all()


    def inference_simple(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding,self.input_x)
        lstm_fw_cell = rnn.GRUCell(self.hidden_size)
        lstm_bw_cell = rnn.GRUCell(self.hidden_size)
        ## dropout to rnn
        #if self.is_training:
        #    lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout_keep_prob)
        #    lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout_keep_prob)
        outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_words, sequence_length=self.length, dtype=tf.float32)
        self.hidden_state = tf.concat(outputs, axis=2)
        ## attention ##
        hidden_state_ = tf.reshape(self.hidden_state, shape=[-1, self.hidden_size *2])
        u = tf.nn.tanh(tf.matmul(hidden_state_,self.W_w_attention_word,)+self.W_b_attention_word)
        u = tf.reshape(u, shape=[self.batch_size, -1, self.hidden_size * 2])
        uv = tf.multiply(u, self.context_vecotor_word)
        uv = tf.reduce_sum(uv, axis=2)
        ## Mask ##
        uv+=(1.0 - tf.cast(self.input_mask, tf.float32))*-10000.0
        self.attention = tf.nn.softmax(uv-tf.reduce_max(uv,-1,keepdims=True))

        ## output ##
        hidden_new = self.hidden_state*tf.expand_dims(self.attention,-1) 
        sentence_representation = tf.reduce_sum(hidden_new, 1) 

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(sentence_representation,keep_prob=self.dropout_keep_prob)

        # 5. logits
        with tf.name_scope("output"):
                logits = tf.matmul(self.h_drop, self.W_projection) + self.b_projection

        #tag head logits
        if self.tag_flag:
            with tf.name_scope("tag_output"):
                hidden_state_tag = tf.reshape(hidden_new, shape=[-1,self.hidden_size*2])
                self.tag_drop = tf.nn.dropout(hidden_state_tag, keep_prob=self.dropout_keep_prob)
                self.tag_logitss = tf.matmul(self.tag_drop, self.W_tag) + self.b_tag
                tag_logits = tf.reshape(self.tag_logitss, [self.batch_size, -1, self.num_tags])
                return logits, tag_logits 
                
        return logits
            
        
    def loss(self, l2_lambda=0.0001):
        with tf.name_scope("loss"):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def loss_multilabel(self, l2_lambda=0.00001*10):
        with tf.name_scope("loss"):
            # input: `logits` and `labels` must have the same shape `[batch_size, num_classes]
            if self.loss_number==0:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
            elif self.loss_number==1:
                losses = tf.nn.weighted_cross_entropy_with_logits(targets=self.input_y_multilabel,logits=self.logits,pos_weight=2)
            else:
                losses = tf.losses.mean_pairwise_squared_error(labels=self.input_y_multilabel, predictions=tf.sigmoid(self.logits))
            
            # losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)

            print("sigmoid_cross_entropy_with_logits.losses:", losses)
            if self.loss_number==0 or self.loss_number==1:
                losses = tf.reduce_sum(losses, axis=1)
            loss = tf.reduce_mean(losses)

            #for tag head
            log_probs = tf.nn.log_softmax(self.tag_logits, axis=-1)
            one_hot_labels = tf.one_hot(self.input_tag, depth=self.num_tags, dtype=tf.float32)
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1) 
            loss_tag = tf.reduce_mean(per_example_loss)
            #loss_tag = tf.reduce_sum(per_example_loss)
            # loss for tag and classier
            loss = loss + loss_tag     
            #l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            #loss = loss + l2_losses
        return loss

    def loss_count(self):
        num_bins = self.num_count
        sentence_level_output = tf.stop_gradient(self.h_drop)
        cnt_h1 = tf.layers.dense(sentence_level_output, 1024, activation=tf.nn.relu)
        cnt_h2 = tf.layers.dense(cnt_h1, 512, activation=tf.nn.relu)
        cnt_h3 = tf.layers.dense(cnt_h2, 256, activation=tf.nn.relu)
        lcnt = tf.layers.dense(cnt_h3, self.num_count, activation=tf.nn.relu)
        predictions_count = tf.argmax(lcnt, axis=1, name='prediction_count')
        label_count = tf.reduce_sum(self.input_y_multilabel,1)
        tails = num_bins*tf.ones_like(label_count)
        bins = tf.where(label_count > num_bins, tails, label_count)
        labels = bins -1
        labels = tf.cast(labels, tf.int64)

        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lcnt, labels=labels)
        lcnt_loss = tf.reduce_mean(xent, name ='count_loss')

        correct_pred = tf.equal(labels,predictions_count)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return predictions_count, lcnt_loss, acc


    def train(self):
        """based on the loss, use SGD to update parameter"""
        self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf_contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=self.decay_learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return train_op

    def train_count(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,
                                                   self.decay_rate, staircase=True)
        self.learning_rate_=learning_rate
        count_train_op = tf_contrib.layers.optimize_loss(self.lcnt_loss, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam",clip_gradients=self.clip_gradients)
        return count_train_op

    def instantiate_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding_projection"):
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],initializer=self.initializer)
            if self.char_attention_flag:
                self.CharEmbedding = tf.get_variable("CharEmbedding", shape=[self.char_size, self.embed_size],initializer=self.initializer)
                self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 4, self.num_classes],initializer=self.initializer)
            else:
                self.W_projection = tf.get_variable("W_projection", shape=[self.hidden_size * 2, self.num_classes],initializer=self.initializer)
            self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

        
        # for tag head
        if self.tag_flag:
            self.W_tag = tf.get_variable("W_tag", shape=[self.hidden_size*2, self.num_tags], initializer=self.initializer)
            self.b_tag = tf.get_variable("b_tag", shape=[self.num_tags])
            

        with tf.name_scope("attention"):
            self.W_w_attention_word = tf.get_variable("W_w_attention_word",shape=[self.hidden_size * 2, self.hidden_size * 2],initializer=self.initializer)
            self.W_b_attention_word = tf.get_variable("W_b_attention_word", shape=[self.hidden_size * 2])
            self.context_vecotor_word = tf.get_variable("what_is_the_informative_word", shape=[self.hidden_size * 2],initializer=self.initializer)
            if self.char_attention_flag:
                self.W_w_attention_char = tf.get_variable("W_w_attention_char",shape=[self.hidden_size * 2, self.hidden_size * 2],initializer=self.initializer)
                self.W_b_attention_char = tf.get_variable("W_b_attention_char", shape=[self.hidden_size * 2])
                self.context_vecotor_char = tf.get_variable("what_is_the_informative_char", shape=[self.hidden_size * 2],initializer=self.initializer)


