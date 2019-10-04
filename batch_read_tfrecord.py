#-*-coding=utf-8-*-


import tensorflow as tf



class SegBatcher(object):
    def __init__(self, record_file_name, batch_size, num_epochs=None):
        self._batch_size = batch_size
        self.num_epochs = num_epochs
        self.next_batch_op = self.input_pipeline(record_file_name, self._batch_size, self.num_epochs)

    def example_parser(self, filename_queue):
        reader = tf.TFRecordReader()
        key, record_string = reader.read(filename_queue)
        features = {
            'terms': tf.FixedLenSequenceFeature([], tf.int64),
            'chars': tf.FixedLenSequenceFeature([], tf.int64),
            'labels': tf.FixedLenSequenceFeature([], tf.float32),
            'tags': tf.FixedLenSequenceFeature([], tf.int64)
        }
        _, example = tf.parse_single_sequence_example(serialized=record_string, sequence_features=features)
        terms = example['terms']
        chars = example['chars']
        labels = example['labels']
        tags = example['tags']
        return terms, chars, labels, tags

    def input_pipeline(self, filenames, batch_size,  num_epochs=None):
        filename_queue = tf.train.string_input_producer([filenames], num_epochs=num_epochs, shuffle=True)
        terms, chars, labels, tags = self.example_parser(filename_queue)
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 12 * batch_size
        print ('starting to begaining')
        next_batch = tf.train.batch([terms, chars, labels, tags], batch_size=batch_size, capacity=capacity, dynamic_pad=True, allow_smaller_final_batch=True)
        print ('ending to .......')
        return next_batch
