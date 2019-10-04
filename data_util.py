# -*- coding: utf-8 -*-
import codecs
import numpy as np
#load data of zhihu
#import word2vec
import os
import pickle
import codecs
#from tflearn.data_utils import pad_sequences

def load_cid(cid3_file):
    label_name = {}
    with open(cid3_file) as f:
        for line in f:
            line_list = line.strip().split('\t')
            label_name[line_list[1]] = line_list[2]
    return label_name

def load_tag(tag_file):
    tag_name = {}
    with open(tag_file) as f:
        for line in f:
            line_list = line.strip().split('\t')
            tag_name[str(line_list[1])]= line_list[0]
    return tag_name

def load_test(test_file,vocabulary_word2index, vocabulary_char2index):
    testX =[]
    testXChar = []
    lines = []
    with open(test_file) as f:
        for line in f:
            line_list = line.strip('\r\n').split('\t')
            if len(line_list) < 2:
                print(line)
                continue
            x = [vocabulary_word2index.get(e,1) for e in line_list[1].split(',')]
            char = [vocabulary_char2index.get(e,1) for e in list(line_list[0].replace(" ",''))]
            testX.append(x)
            testXChar.append(char)
            lines.append(line.strip())
    return testX, testXChar, lines


def create_term(term_file):
    term2idx = {}
    idx2term = {}
    with open(term_file ) as f:
        for line in f:
            line = line.replace('\r\n','\n')
            line = line.strip('\n')
            if line:
                tokens = line.split('\t')
                if len(tokens) !=2:
                    print (line)
                term2idx[tokens[0]] = int(tokens[1])
                idx2term[int(tokens[1])] = tokens[0]
    return term2idx, idx2term

def create_label(label_file):
    label2idx = {}
    idx2label = {}
    with open(label_file) as f:
        for line in f:
            line = line.replace('\r\n','\n')
            line = line.strip()
            if line:
                tokens = line.split('\t')
                label2idx[tokens[1]] = int(tokens[0]) -1
                idx2label[int(tokens[0])-1] = tokens[1]
    return label2idx, idx2label
