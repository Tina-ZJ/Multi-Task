#coding:utf-8
###################################################################
import os
import logging
import logging.handlers
import tensorflow as tf

#version
SERVER_NAME = 'ner'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = BASE_DIR + '/data'
MODEL_DIR = BASE_DIR + '/output'


#files path
STOPWORD_FILE = DATA_DIR + '/stopword_data/stop_symbol'
LABEL_FILE = MODEL_DIR + '/labels.tsv'
LABEL_MAP_FILE = MODEL_DIR + '/label_map'
LABEL_CID_MAP_FILE = MODEL_DIR + '/label_cid3_map'
VOCAB_FILE = MODEL_DIR + '/vocab.txt'
BERT_CONFIG_FILE = MODEL_DIR + '/albert_config_tiny.json'
CHECKPOINT_DIR = MODEL_DIR + '/checkpoints'


LOG_DIR = BASE_DIR + '/log/'
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(lineno)s]%(filename)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)
log_handler = logging.handlers.TimedRotatingFileHandler(filename=LOG_DIR + 'ner.log', when='D', interval=1, backupCount=10)
log_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logging.getLogger('').addHandler(log_handler)
