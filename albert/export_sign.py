#coding:utf-8
###################################################
import os
import sys
import json
import shutil
import tensorflow as tf
import modeling 



from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.model_fn import EstimatorSpec
from tensorflow.python.estimator.export.export_output import PredictOutput

from run_sequencelabeling import create_model
from preprocess import bert_data_utils
from setting import *

SIGNATURE_NAME='albert_multi_task'

def model_fn_builder(bert_config, num_labels, num_cid3_labels, init_checkpoint,
                                         use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = tf.ones(tf.shape(input_ids), dtype=tf.int32)
        label_cid3_ids = tf.ones(tf.shape(input_ids), dtype=tf.int32)
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(input_ids)[0], dtype=tf.float32)

        input_ids = tf.placeholder_with_default(input_ids, shape=[None, input_ids.shape[1]], name='input_ids')
        input_mask = tf.placeholder_with_default(input_mask, shape=[None, input_mask.shape[1]], name='input_mask')
        segment_ids = tf.placeholder_with_default(segment_ids, shape=[None, segment_ids.shape[1]], name='segment_ids')

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits,logits_cid3, pred_label_ids, probabilities, acc, predictions, char_weight, precision, recall, f1) = create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                label_cid3_ids, num_labels, num_cid3_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("    name = %s, shape = %s%s", var.name, var.shape,
                                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = tf.estimator.EstimatorSpec(
                        mode=mode,
                        predictions={ "predictions": predictions,
                                      "char_weight": char_weight,
                                      "pred_label_ids": pred_label_ids},
                        export_outputs={SIGNATURE_NAME: PredictOutput({"pred_label_ids": pred_label_ids,
                                                                       "char_weight": char_weight,
                                                                       "predictions": predictions})})
        return output_spec

    return model_fn



def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders
    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
    segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')

    receiver_tensors = {'input_ids': input_ids,
                        'input_mask': input_mask,
                        'segment_ids': segment_ids}

    features = {'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids}

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


if __name__ == '__main__':
    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG_FILE)
    label2idx, idx2label = bert_data_utils.read_ner_label_map_file(LABEL_MAP_FILE)
    label_cid2idx, idx2label_cid = bert_data_utils.read_ner_label_map_file(LABEL_CID_MAP_FILE)
    num_labels = len(label2idx)
    num_cid3_labels = len(label_cid2idx)
    
    cp_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=num_labels,
            num_cid3_labels=num_cid3_labels,
            init_checkpoint=cp_file,
            use_one_hot_embeddings=False) 


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.log_device_placement = False
    batch_size = 1
    export_dir = CHECKPOINT_DIR
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=MODEL_DIR, config=RunConfig(session_config=config),
                                                             params={'batch_size': batch_size})

    estimator.export_saved_model(export_dir, serving_input_receiver_fn, checkpoint_path=cp_file)

