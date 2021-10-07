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
from model import Input
from model import Model

def parse_args():
    parser=argparse.ArgumentParser("LSTM experiments for translation tasks.")
    
    parser.add_argument("--config",type=str,default='config.yaml',help="configuration file")
    parser.add_argument("--save_dir",type=str,default="train_dir/",help="directory in which training state and model should be saved")
    parser.add_argument("--data_path",type=str,default="../my_lstm/",help="path in which the raw data is located")
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

def load_data(data_path,logger):
    train_path=os.path.join(data_path,"ptb.train.txt")
    valid_path=os.path.join(data_path,"ptb.valid.txt")
    test_path=os.path.join(data_path,"ptb.test.txt")

    with open(train_path) as f1,open(valid_path) as f2,open(test_path) as f3:
        seq_train=f1.read().replace('\n','<eos>').split(' ')
        seq_valid=f2.read().replace('\n','<eos>').split(' ')
        seq_test=f3.read().replace('\n','<eos>').split(' ')
    

    seq_train=list(filter(None,seq_train))
    seq_valid=list(filter(None,seq_valid))
    seq_test=list(filter(None,seq_test))
    
    size_train,size_valid,size_test = len(seq_train),len(seq_valid),len(seq_test)
    logger.info("The size of seq_train, seq_valid and seq_test is {}, {} and {}".format(size_train,size_valid,size_test))

    vocab_train=set(seq_train)
    vocab_valid=set(seq_valid)
    vocab_test=set(seq_test)
    logger.info("The size of vocab_train, vocab_valid and vocab_test is {}, {} and {}".format(len(vocab_train),len(vocab_valid),len(vocab_test)))
   
    vocabulary=sorted(vocab_train)
    word2id={w:i for i,w in enumerate(vocabulary)}
    id2word={i:w for i,w in enumerate(vocabulary)}
    
    train_data=np.array([word2id[word] for word in seq_train])
    valid_data=np.array([word2id[word] for word in seq_valid])
    test_data=np.array([word2id[word] for word in seq_test])
    
    return train_data,valid_data,test_data,word2id,id2word

def train(arglist):

    #***************************************************************************
    #-------------------------------Set Up--------------------------------------
    #                                                                          *
    #***************************************************************************
    arglist.save_dir=Path(arglist.save_dir)
    if not arglist.save_dir.exists():
        os.makedirs(arglist.save_dir)

    now=datetime.datetime.now()
    log_file=arglist.save_dir/now.strftime("%Y%m%d_ %H%M%S.log")
    logger=init_logger(log_file)
    logger.info("date {}".format(datetime.datetime.now()))
    logger.info("device {}".format(tf.test.gpu_device_name()))
    logger.info("Tensorflow version {}".format(tf.__version__))

    config=yaml.load(open(arglist.config,'r'),Loader=yaml.FullLoader)
    config=EasyDict(config)
    batch_size=config.train.batch_size
    num_steps=config.train.num_steps
    hidden_size=config.train.hidden_size
    num_layers=config.train.num_layers    
    learning_rate=config.train.learning_rate
    max_lr_epoch=config.train.max_lr_epoch
    lr_decay=config.train.lr_decay
    num_epoches=10
    train_data,valid_data,test_data,word2id,id2word=load_data(arglist.data_path,logger)    
    
    #***************************************************************************
    #-------------------------------Model---------------------------------------
    #                                                                          *
    #***************************************************************************

    training_input=Input(batch_size=batch_size,num_steps=num_steps,data=train_data)
    model=Model(arglist.task_id,
                training_input,
                is_training=True,
                batch_size=batch_size,
                num_steps=num_steps,
                hidden_size=hidden_size,
                vocab_size=len(word2id),
                num_layers=num_layers)
    init_op=tf.global_variables_initializer()
 
    orig_decay=lr_decay
    model_save_name='my_lstm'
    with tf.Session() as sess:
        sess.run([init_op])
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        saver=tf.train.Saver()
	
        for epoch in range(arglist.num_episodes):
            new_lr_decay=orig_decay**max(epoch+1-max_lr_epoch,0.0)
            model.assign_lr(sess,learning_rate*new_lr_decay)
            current_state=np.zeros((num_layers,2,batch_size,model.hidden_size))
            for step in range(training_input.epoch_size):
                if step%50!=0:
                    cost,_,current_state=sess.run([model.cost,model.train_op,model.state],feed_dict={model.init_state:current_state})
                else:
                    cost,_,current_state,acc=sess.run([model.cost,model.train_op,model.state,model.accuracy],feed_dict={model.init_state:current_state})		
                    print("Epoch: {}, Step: {}, Cost: {:.3f}, Accuracy: {:.3f}".format(epoch,step,cost,acc))

            saver.save(sess,arglist.data_path+"\\"+model_save_name,global_step=epoch)
        saver.save(sess,arglist.data_path+"\\"+model_save_name+"_final")
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    arglist=parse_args()
    train(arglist)

































