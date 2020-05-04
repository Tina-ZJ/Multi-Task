#-*-coding=utf-8-*-


import codecs
import numpy as np



def get_cid_index(filename):
    cid_index = {}
    with open(filename) as f:
        for line in f:
            line = line.replace('\r\n', '\n')
            line = line.strip('\n')
            if line:
                tokens = line.split('\t')
                cid_index[int(tokens[1])] = int(tokens[0])
    return cid_index

def get_index_cid(filename):
    index_cid = {}
    with open(filename) as f:
        for line in f:
            line = line.replace('\r\n', '\n')
            line = line.strip('\n')
            if line:
                tokens = line.split('\t')
                index_cid[int(tokens[0])] = int(tokens[1])
    return index_cid

def get_index_cid_name(filename):
    index_cid_name = {}
    with open(filename) as f:
        for line in f:
            line = line.replace('\r\n', '\n')
            line = line.strip('\n')
            if line:
                tokens = line.split('\t')
                index_cid_name[int(tokens[0])] = tokens[2]
    return index_cid_name

def get_term_index(filename):
    index_term = {}
    term_index = {}
    with open(filename) as f:
        for line in f:
            line = line.replace('\r\n', '\n')
            line = line.strip('\n')
            if line:
                tokens = line.split('\t')
                if len(tokens)!=2:
                    print(line)
                index_term[int(tokens[1])] = tokens[0]
                term_index[tokens[0]] = int(tokens[1])
    return index_term, term_index

def get_one_hot_label(labels, class_num):
    one_hot_labels = []
    for label in labels:
        one_hot_label = [0.0] * class_num
        for lab in label:
            if int(lab) != 0:
                one_hot_label[int(lab)-1] = 1.0
        one_hot_labels.append(one_hot_label)
    return one_hot_labels

