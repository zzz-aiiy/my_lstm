import tensorflow as tf
import numpy as np

def batch_producer(raw_data,batch_size,num_steps):
    raw_data=tf.convert_to_tensor(raw_data,name='raw_data',dtype=tf.int32)

    data_len=tf.size(raw_data)
    batch_num=data_len//batch_size
    data=tf.reshape(raw_data[0:batch_size*batch_num],[batch_size,batch_num])
    
    epoch_size=(batch_num-1)//num_steps

    i=tf.train.range_input_producer(epoch_size,shuffle=False).dequeue()
    x=data[:,i*num_steps:(i+1)*num_steps]
    x.set_shape([batch_size,num_steps])
    y=data[:,i*num_steps+1:(i+1)*num_steps+1]
    y.set_shape([batch_size,num_steps])
    
    return x,y

class Input(object):
    def __init__(self,batch_size,num_steps,data):
        self.batch_size=batch_size
        self.num_steps=num_steps
        self.epoch_size=((len(data)//batch_size)-1)//num_steps
        self.input_data,self.targets=batch_producer(data,batch_size,num_steps)

class Model(object):
    def __init__(self,task_id,input,is_training,batch_size,num_steps,hidden_size,vocab_size,num_layers,dropout=0.5,init_scale=0.05):
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
        self.new_lr=tf.placeholder(tf.float32,shape=[])

        self.optimize_func()	

    def word_embedding(self):
        with tf.device("/cpu:0"):
            embedding=tf.Variable(tf.random_uniform([self.vocab_size,self.hidden_size],-self.init_scale,self.init_scale),name='Embedding')
            inputs=tf.nn.embedding_lookup(embedding,self.input_obj.input_data)
        
        if self.is_training and self.dropout<1:
            inputs=tf.nn.dropout(inputs,self.dropout)
        
        return inputs
    
    def network(self,inputs):
        
        state_per_layer_list=tf.unstack(self.init_state,axis=0)
        rnn_tuple_state=tuple([tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0],state_per_layer_list[idx][1]) for idx in range(self.num_layers)])  
       
        cell=tf.contrib.rnn.LSTMCell(self.hidden_size,forget_bias=1.0)
        if self.is_training and self.dropout<1:
            cell=tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=self.dropout)

        if self.num_layers>1:
            cell=tf.contrib.rnn.MultiRNNCell([cell for _ in range(self.num_layers)],state_is_tuple=True)
        
        output,state=tf.nn.dynamic_rnn(cell,inputs,dtype=tf.float32,initial_state=rnn_tuple_state)
        return output,state

    def optimize_func(self):
        
        self.embedding=self.word_embedding()
        self.output,self.state=self.network(self.embedding)
        self.output=tf.reshape(self.output,[-1,self.hidden_size])
        
        softmax_w=tf.Variable(tf.random_uniform([self.hidden_size,self.vocab_size],-self.init_scale,self.init_scale))
        softmax_b=tf.Variable(tf.random_uniform([self.vocab_size],-self.init_scale,self.init_scale))
        logits=tf.nn.xw_plus_b(self.output,softmax_w,softmax_b)
        logits=tf.reshape(logits,[self.batch_size,self.num_steps,self.vocab_size])
       
        loss=tf.contrib.seq2seq.sequence_loss(
                logits,
                self.input_obj.targets,
                tf.ones([self.batch_size,self.num_steps],tf.float32),
                average_across_timesteps=False,
                average_across_batch=True)
 
        self.cost=tf.reduce_sum(loss)
	
        self.softmax_out=tf.nn.softmax(tf.reshape(logits,[-1,self.vocab_size]))
        self.predict=tf.cast(tf.argmax(self.softmax_out,axis=1),tf.int32)
        correct_prediction=tf.equal(self.predict,tf.reshape(self.input_obj.targets,[-1]))
        self.accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        if not self.is_training:
            return 
	
        self.learning_rate=tf.Variable(0.0,trainable=False)
	
        tvars=tf.trainable_variables()	
        grads,_=tf.clip_by_global_norm(tf.gradients(self.cost,tvars),5)
        optimizer=tf.train.GradientDescentOptimizer(self.learning_rate)

        self.train_op=optimizer.apply_gradients(
                zip(grads,tvars),
                global_step=tf.contrib.framework.get_or_create_global_step())

	
        self.lr_update=tf.assign(self.learning_rate,self.new_lr)

    def assign_lr(self,session,lr_value):
        session.run(self.lr_update,feed_dict={self.new_lr:lr_value})





































    



















