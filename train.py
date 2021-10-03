import tensorflow as tf
import numpy as np
import logging
import time
import math
import datetime
import collections
import argparse
import yaml
from easydict import EasyDict
import os
from pathlib import Path

def parse_args():
    parser=argparse.ArgumentParser("LSTM experiments for translation tasks.")
    
    parser.add_argument("--config",type=str,default='config.yaml',help="configuration file")
    parser.add_argument("--save-dir",type=str,default="train_dir/",help="directory in which training state and model should be saved")
    parser.add_argument("--batch-size",type=int,default=0,help="batch size")
    parser.add_argument("--seq-len",type=int,default=0,help="the length of a sequence")

    return parser.parse_args()

def init_logger(log_file):
    logger=logging.getLogger()
    logger.setLevel(logging.INFO)
    
    fd=logging.FileHandler(log_file,mode='w')
    fd.setLevel(logging.INFO)
    
    logger.addHandler(fd)
    logger.addHandler(logging.StreamHandler())
    
    return logger

#MIT License - Copyright (c) 2018 tmatha
def features_labels(data_array,batch_size,seq_len,batch_first=True):
    if len(data_array.shape)!=1:
        raise ValueError('Expected 1-d data data array, '
                         'instead data array shape is {}'.format(data_array.shape))


def train(arglist):
    #arglist.save_dir=os.path.abspath(arglist.save_dir)
    #if not arglist.save_dir.exists():
    #    os.makedirs(arglist.save_dir)

    config=yaml.load(open(arglist.config,'r'),Loader=yaml.FullLoader)
    config=EasyDict(config)

    now=datetime.datetime.now()
    log_file=arglist.save_dir+'/'+now.strftime("%Y%m%d_%H%M%S.log")
    logger=init_logger(log_file)
    logger.info("date {}".format(datetime.datetime.now()))
    logger.info("device {}".format(tf.test.gpu_device_name()))
    logger.info("Tensorflow version {}".format(tf.__version__))


    with open('ptb.train.txt') as f1,open('ptb.valid.txt') as f2,open('ptb.test.txt') as f3:
        seq_train=f1.read().replace('\n','<eos>').split(' ')
        seq_valid=f2.read().replace('\n','<eos>').split(' ')
        seq_test=f3.read().replace('\n','<eos>').split(' ')

    seq_train=list(filter(None,seq_train))
    seq_valid=list(filter(None,seq_valid))
    seq_test=list(filter(None,seq_test))

    size_train=len(seq_train)
    size_valid=len(seq_valid)
    size_test=len(seq_test)
    logger.info('size_train:{}, size_valid:{}, size_test:{}'.format(size_train,size_valid,size_test))

    vocab_train=set(seq_train)
    vocab_valid=set(seq_valid)
    vocab_test=set(seq_test)
    logger.info('vocab_train:{}, vocab_valid:{}, vocab_test:{}'.format(len(vocab_train),len(vocab_valid),len(vocab_test)))

    vocab_train=sorted(vocab_train)
    word2id={w:i for i,w in enumerate(vocab_train)}
    id2word={i:w for i,w in enumerate(vocab_train)}

    ids_train=np.array([word2id[word] for word in seq_train],copy=False,order='C')
    ids_valid=np.array([word2id[word] for word in seq_valid],copy=False,order='C')
    ids_test=np.array([word2id[word] for word in seq_test],copy=False,order='C')


if __name__ == '__main__':
    arglist=parse_args()
    train(arglist)







