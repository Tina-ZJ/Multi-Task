# -*- coding:utf8 -*-
import sys


def cdf(file1, file2):
    f = open(file1)
    f2 = open(file2,'w+')
    for line in f:
        cid3 = list()
        cid3_new = list()
        name_new = list()
        all_scores = 0.0
        cdf_scores = 0.0
        score = list()
        terms = line.strip().split('\t')
        name = terms[3].split(',')
        for x in terms[2].split(','):
            cid3.append(x.split(':')[0])
            score.append(x.split(':')[1])
            all_scores+=float(x.split(':')[1])
        if all_scores==0.0:
            all_scores=0.001
        for c,s,n in zip(cid3,score,name):
            t = float(s)/all_scores
            if cdf_scores<0.9 :
                cdf_scores+=t
                cid3_new.append(c+':'+str(t))
                name_new.append(n)
            elif float(s)>=0.13:
                cdf_scores+=t
                cid3_new.append(c+':'+str(t))
                name_new.append(n)
        f2.write(terms[0]+'\t'+terms[1]+'\t'+','.join(cid3_new)+'\t'+','.join(name_new)+'\t'+terms[4]+'\n')

if __name__=='__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    cdf(file1,file2) 
            
