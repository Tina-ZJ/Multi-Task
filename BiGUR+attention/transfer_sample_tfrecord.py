#-*-coding=utf-8-*-


import re
import codecs
import tensorflow as tf
import common
import argparse




def sample_to_tfrecord(sample_file, tfrecord_file, term_index, char_index, tag_index, cid_name_file):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    num = 0
    cid_index = common.get_cid_index(cid_name_file)
    with codecs.open(sample_file, errors="ignore", encoding='utf8') as f:
        for i, line in enumerate(f):
            num += 1
            example = tf.train.SequenceExample()
            fl_terms = example.feature_lists.feature_list["terms"]
            fl_chars = example.feature_lists.feature_list["chars"]
            fl_labels = example.feature_lists.feature_list["labels"]
            fl_tags = example.feature_lists.feature_list["tags"]
            line = line.replace('\r\n', '\n')
            line = re.sub('[ ]+', ' ', line)
            tokens = line.strip('\n ').split('\t')
            if len(tokens) == 3 and tokens[0].strip()!='' and tokens[0].strip()!='NULL':
                query = tokens[1].strip()
                chars = tokens[0].strip()
                label = tokens[2].strip()
                if len(query) == 0 or len(label) == 0:
                    continue
                term_list = query.split(',')
                char_list = list(chars)
                label_list = label.split(',')
                for lab in label_list:
                    index = cid_index[int(lab)]
                    fl_labels.feature.add().float_list.value.append(int(index))
                for terms in term_list:
                    term, tag = terms.split('|')
                    if term.strip()=='':
                        continue
                    if term in term_index:
                        fl_terms.feature.add().int64_list.value.append(term_index[term])
                    else:
                        fl_terms.feature.add().int64_list.value.append(1)
                    if tag in tags_index:
                        fl_tags.feature.add().int64_list.value.append(tag_index[tag])
                    else:
                        fl_tags.feature.add().int64_list.value.append(0)
                for char in char_list:
                    if char in char_index:
                        fl_chars.feature.add().int64_list.value.append(char_index[char])
                    else:
                        fl_chars.feature.add().int64_list.value.append(1)
            writer.write(example.SerializeToString())
            if i % 10000 == 0 and i != 0:
                print('{i} line sample transfer succeed....'.format(i=i))
                writer.flush()
    return num


def save_sample_size(train_sample_size, dev_sample_size, sample_size_file):
    with codecs.open(sample_size_file, 'w', encoding='utf8') as f:
        f.write("train_sample_size:%d\n" % (int(train_sample_size)))
        f.write("dev_sample_size:%d\n" % (int(dev_sample_size)))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid_name_file", nargs='?', default='./data/cid3_name.txt')
    parser.add_argument("--train_sample_file", nargs='?', default='./data/train_sample.txt')
    parser.add_argument("--dev_sample_file", nargs='?', default='./data/dev_sample.txt')
    parser.add_argument("--sample_size_file", nargs='?', default='./data/sample_size.txt')
    parser.add_argument("--term_index_file", nargs='?', default='./data/term_index.txt')
    parser.add_argument("--char_index_file", nargs='?', default='./data/char_index.txt')
    parser.add_argument("--tags_index_file", nargs='?', default='./data/tags_index.txt')

    args = parser.parse_args()
    train_tfrecord_file = args.train_sample_file[:-4] + '.tfrecord'
    dev_tfrecord_file = args.dev_sample_file[:-4] + '.tfrecord'
    _, term_index = common.get_term_index(args.term_index_file)
    _, char_index = common.get_term_index(args.char_index_file)
    _, tags_index = common.get_term_index(args.tags_index_file)
    train_sample_size = sample_to_tfrecord(args.train_sample_file, train_tfrecord_file, term_index, char_index, tags_index, args.cid_name_file)
    print('train sample transfer to tfrecord succeed....')
    dev_sample_size = sample_to_tfrecord(args.dev_sample_file, dev_tfrecord_file, term_index, char_index, tags_index, args.cid_name_file)
    print('dev sample transfer to tfrecord succeed....')
    save_sample_size(train_sample_size, dev_sample_size, args.sample_size_file)
    print('save sample size to file succeed....')
