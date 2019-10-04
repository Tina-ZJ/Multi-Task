#-*-coding=utf-8-*-


import random
import argparse
import codecs





def split_sample(all_sample_file, train_sample_file, dev_sample_file, train_sample_ratio):
    all_sample = []
    with open(all_sample_file) as f:
        for line in f:
            all_sample.append(line.strip())
    random.shuffle(all_sample)
    all_sample_num = len(all_sample)
    train_sample_num = int(all_sample_num * train_sample_ratio)
    train_sample = all_sample[:train_sample_num]
    dev_sample = all_sample[train_sample_num:]
    with open(train_sample_file, 'w') as f:
        for sample in train_sample:
            f.write(sample + '\n')
    with open(dev_sample_file, 'w') as f:
        for sample in dev_sample:
            f.write(sample + '\n')
    print('split sample succeed....')







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_sample_file', nargs='?', default='./data/all_sample.txt')
    parser.add_argument('--train_sample_file', nargs='?', default='./data/train_sample.txt')
    parser.add_argument('--dev_sample_file', nargs='?', default='./data/dev_sample.txt')
    parser.add_argument('--train_sample_ratio', nargs='?', type=float, default=0.98)
    args = parser.parse_args()
    split_sample(args.all_sample_file, args.train_sample_file, args.dev_sample_file, args.train_sample_ratio)
