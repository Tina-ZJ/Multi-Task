import tensorflow as tf
import numpy as np
import fileinput
import os
import pickle as pick


def extract_parameter(modelPath):
    nw = open(os.path.join(modelPath, "weight.names"), 'w')
    out_f = open(os.path.join(modelPath, "model.w"), "w")
    saver = tf.train.import_meta_graph(os.path.join(modelPath, "model.ckpt-0.meta"))
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        with tf.device("/cpu:0"):
            saver.restore(sess, os.path.join(modelPath, "model.ckpt-0"))
            for var in tf.trainable_variables():
                data = sess.run(var)
                if len(data.shape) == 1:
                    data = data.reshape(1, data.shape[0])
                name = str(var.name).split("/")[-1]
                nw.write(name+" " + str(data.shape) + "\n")
                np.savetxt(out_f, data, fmt='%s', newline='\n')
    out_f.close()
    nw.flush()
    nw.close()


def save_pkl(modelPath):
    parames = dict()
    sum = 0
    saver = tf.train.import_meta_graph(os.path.join(modelPath, "model_ckpt-18.meta"))
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        saver.restore(sess, os.path.join(modelPath, "model"))
        for var in tf.trainable_variables():
            sum += 1
            data = sess.run(var)
            name = var.name
            if sum == 1 or sum == len(tf.trainable_variables()) or sum == len(
                    tf.trainable_variables()) - 1 or sum == len(tf.trainable_variables()) - 2:
                continue
            parames[name] = data
        f = file(os.path.join(modelPath, "params.pkl"), 'wb')
        pick.dump(parames, f, True)
        f.close()


def read_pkl(modelPath):
    f2 = file(os.path.join(modelPath, "params.pkl"), 'rb')
    result = pick.load(f2)
    f2.close()
    return result

if __name__ == "__main__":
    extract_parameter("./checkpoint/HAN_300d_one")
