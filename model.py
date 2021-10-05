import tensorflow as tf
import numpy as np

def batch_producer(raw_data,batch_size,num_steps):
    raw_data=tf.convert_to_tensor(raw_data,name='raw_data',dtype=tf.int32)

    data_len=tf.size(raw_data)
    batch_num=data_len//batch_size
    data=tf.reshape(raw_data[0:batch_size*batch_num],[batch_num,batch_size])
    
    epoch_size=(batch_size-1)//num_steps

    i=tf.train.range_input_producer(epoch_size,shuffle=False).dequeue()
    x=data[:,i*num_steps,(i+1)*num_steps]
    x.set_shape([batch_size,num_steps])
    y=data[:,i*num_steps+1,(i+1)*num_steps+1]
    y.set_shape([batch_size,num_steps])
    
    return x,y

class Input(object):
    def __init__(self,batch_size,num_steps,data):
        self.batch_size=batch_size
        self.num_steps_num_steps
        self.epoch_size=((len(data)//batch_size)-1)//num_steps
        self.input_data,self.targets=batch_producer(data,batch_size,num_steps)

class Model(object):
    def __init__(self,task_id,input,is_training,hidden_size,vocab_size,num_layers,dropout=0.5,init_scale=0.05):
        self.is_training=is_training
        self.input_obj=input
        self.batch_size=batch_size
        self.num_steps=num_steps
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.num_layers=num_layers
        self.dropout=dropout
        self.init_scale=init_scale
        self.task_id=task_id
        
        self.init_state=tf.placeholder(tf.float32,[self.num_layers,2,self.batch_size,self.hidden_size])
    
        self.embedding=self.word_embedding()

    def word_embedding(self):
        with tf.device("/cpu:0:):
            embedding=tf.Variable(tf.random_uniform([self.vocab_size,self.hidden_size],-self.init_scale,self.init_scale),name='Embedding')
            inputs=tf.nn.embedding_lookup(embedding,self.input_obj.input_data)
        
        if self.is_training and self.dropout<1:
            inputs=tf.nn.dropout(inputs,dropout)
        
        return inputs
    
    

    



















