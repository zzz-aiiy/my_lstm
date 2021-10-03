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
    parser.add_argument("--save_dir",type=str,default="train_dir/",help="directory in which training state and model should be saved")
    parser.add_argument("--task_id",type=int,default=0,help="")
    parser.add_argument("--num_episodes",type=int,default=0,help="the number of episodes")

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
    def fold(used_array):
        shaped_array=np.reshape(used_array,(batch_size,seq_len*steps),order='C')
        
        if batch_first:
            return np.concatenate(np.split(shaped_array,steps,axis=1),axis=0)
        else:
            return np.transpose(shaped_array)
    
    #logger.info(config)
    steps=(data_array.shape[0]-1)//(batch_size*seq_len)
    used=batch_size*seq_len*steps

    features=fold(data_array[:used])
    labels=fold(data_array[1:used+1])

    Data=collections.namedtuple('Data',['features','labels'])
    return Data(features=features,labels=labels),steps 
    
 

def train(arglist):
    
    #***************************************************************************
    #-------------------------------Set Up--------------------------------------
    #                                                                          *
    #***************************************************************************
    arglist.save_dir=Path(arglist.save_dir)
    if not arglist.save_dir.exists():
        os.makedirs(arglist.save_dir)

    config=yaml.load(open(arglist.config,'r'),Loader=yaml.FullLoader)
    config=EasyDict(config)
    batch_size=config.train.batch_size
    seq_len=config.train.seq_len

    now=datetime.datetime.now()
    log_file=arglist.save_dir/now.strftime("%Y%m%d_%H%M%S.log")
    logger=init_logger(log_file)
    logger.info("date {}".format(datetime.datetime.now()))
    logger.info("device {}".format(tf.test.gpu_device_name()))
    logger.info("Tensorflow version {}".format(tf.__version__))

    #***************************************************************************
    #-------------------------------Loading Data--------------------------------
    #                                                                          *
    #***************************************************************************
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

    data_train,steps_train=features_labels(ids_train,batch_size,seq_len,batch_first=False)
    data_valid,steps_valid=features_labels(ids_valid,batch_size,seq_len,batch_first=False)
    data_test,steps_test=features_labels(ids_test,batch_size,seq_len,batch_first=False)

    #***************************************************************************
    #------------------------Building Graph-------------------------------------
    #                                                                          *
    #***************************************************************************
    tf.reset_default_graph()
    
    features_placeholder=tf.placeholder(data_train.features.dtype,(None,batch_size))
    labels_placeholder=tf.placeholder(data_train.labels.dtype,(None,batch_size))
    dataset=tf.data.Dataset.from_tensor_slices((features_placeholder,labels_placeholder))
    
    iterator=dataset.batch(seq_len,drop_remainder=True).make_initializable_iterator()
    features,labels=iterator.get_next()
    #TODO
    lr=tf.get_variable('lr',initializer=learning_rate,trainale=False)
    learning_rate_decay=lr.assign(lr*decay)
    
if __name__ == '__main__':
    arglist=parse_args()
    train(arglist)


























