# -*-coding:utf8 -*-
import sys
import codecs

def term(data_file, term_file, char_file, label_file, tag_file):
    f = codecs.open(data_file, encoding='utf8')
    f2 = codecs.open(term_file, 'w', encoding='utf8')
    f3 = codecs.open(label_file, 'w', encoding='utf8')
    f4 = codecs.open(char_file, 'w', encoding='utf8')
    f5 = codecs.open(tag_file, 'w', encoding='utf8')
    words = dict()
    labels = dict()
    tags = dict()
    chars = dict()
    for line in f:
        terms = line.strip().split('\t')
        if len(terms)!=3:
            continue
        char = list(terms[0])
        wts = terms[1].split(',')
        label = terms[2].split(',')
        for wt in wts:
            w,t = wt.split('|')
            if w not in words and w!='':
                words[w] = 0
            if t not in tags and t!='':
                tags[t] = 0
        for c in char:
            if c not in chars and c!='':
                chars[c] = 0
        for la in label:
            if la not in labels :
                labels[la] = 0
    f2.write('<PAD>'+'\t'+str(0)+'\n')
    f2.write('<OOV>'+'\t'+str(1)+'\n')
    f4.write('<PAD>'+'\t'+str(0)+'\n')
    f4.write('<OOV>'+'\t'+str(1)+'\n')
    f5.write('<PAD>'+'\t'+str(0)+'\n')
    for i, x in enumerate(words):
        f2.write(x+'\t'+str(i+2)+'\n')
    for i, c in enumerate(chars):
        f4.write(c+'\t'+str(i+2)+'\n')
    for i, la in enumerate(labels):
        f3.write(str(i+1) +'\t'+str(la)+'\t'+str(la)+'\n')
    for i, t in enumerate(tags):
        f5.write(str(t)+'\t'+str(i+1)+'\n')

if __name__=='__main__':
    data_file = sys.argv[1]
    term_file = sys.argv[2]
    label_file = sys.argv[3]
    char_file = sys.argv[4]
    tag_file = sys.argv[5]
    term(data_file, term_file, char_file, label_file, tag_file) 
        
