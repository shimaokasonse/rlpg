import tensorflow as tf
import numpy as np
from collections import deque
import random

class PolicyNet:
    
    def __init__(self,state_dim,shared_dim,action_dim):
        self.timeStep = 0
        self.state_dim = state_dim
        self.shared_dim = shared_dim
        self.action_dim = action_dim
        self.stateInput = tf.placeholder("float",[None,self.state_dim])
        self.returnInput = tf.placeholder("float",[None])
        self.actionInput = tf.placeholder("float",[None,self.action_dim])
        self.p = self.createNetworkP()
        self.v =self.createNetworkV()
        self.createTrainingStepV()
        self.createTrainingStepP()
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        
        
    def createNetworkP(self):
        Wh1 = self.weight_variable([self.state_dim,self.shared_dim])
        bh1 = self.bias_variable([self.shared_dim])
        Wh2 = self.weight_variable([self.shared_dim,self.shared_dim])
        bh2 = self.bias_variable([self.shared_dim])
        Wp = self.weight_variable([self.shared_dim,self.action_dim])
        bp = self.bias_variable([self.action_dim])
        h1 = tf.nn.relu(tf.matmul(self.stateInput,Wh1)+bh1)
        h2 = tf.nn.relu(tf.matmul(h1,Wh2)+bh2)
        p = tf.nn.softmax(tf.matmul(h2,Wp) + bp)
        return p
    
    def createNetworkV(self):
        Wh1 = self.weight_variable([self.state_dim,self.shared_dim])
        bh1 = self.bias_variable([self.shared_dim])
        Wh2 = self.weight_variable([self.shared_dim,self.shared_dim])
        bh2 = self.bias_variable([self.shared_dim])
        Wv = self.weight_variable([self.shared_dim,1])
        bv = self.bias_variable([1])
        h1 = tf.nn.tanh(tf.matmul(self.stateInput,Wh1)+bh1)
        h2 = tf.nn.tanh(tf.matmul(h1,Wh2)+bh2)
        v = tf.matmul(h2,Wv) + bv
        return v
    
    def createTrainingStepV(self):
        self.lossV = tf.reduce_mean(tf.square(self.returnInput - self.v))
        self.trainStepV = tf.train.GradientDescentOptimizer(0.0001).minimize(self.lossV)
        
    def createTrainingStepP(self):
        self.logp = tf.log(tf.reduce_sum(tf.mul(self.p, self.actionInput),reduction_indices = 1))
        self.lossP = -  tf.reduce_mean(tf.mul(self.logp, (self.returnInput - self.v)))
        self.trainStepP = tf.train.AdamOptimizer(0.001).minimize(self.lossP)
        
    def getAction(self,state):
        prob_a = self.p.eval(feed_dict={self.stateInput:[state]})[0]
        action = np.random.multinomial(1,prob_a)
        return action
    
    def getValue(self,state):
        return self.v.eval(feed_dict={self.stateInput:[state]})[0]
    
    def getActionMax(self,state):
        pass
    
    def trainNetworkV(self,state_batch,return_batch):        
        self.trainStepV.run(feed_dict={self.stateInput:state_batch,
                                       self.returnInput: return_batch})
    
    def trainNetworkP(self,state_batch,action_batch,return_batch):
        self.trainStepP.run(feed_dict={self.stateInput:state_batch,
                                       self.actionInput:action_batch,
                                       self.returnInput: return_batch})
    
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)
        
    def bias_variable(self,shape):
        initial= tf.constant(0.0, shape = shape)
        return tf.Variable(initial)


