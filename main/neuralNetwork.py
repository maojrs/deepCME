import numpy as np
import tensorflow as tf

# Neural network class specialized in solving the Chemical Master Equation (CME)
class neuralNetworkCME:

    def __init__(self, X, num_layers, neurons_per_layer, input_dimension, output_dimension):
        """ Initializes base parameters, architecture, loss function and 
        optimizer for the neural network. It also initializes tensor flow 
        session and variables"""
        
        self.x = X[:,0:1]
        self.t = X[:,1:2]
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        
        self.activation = tf.nn.relu # tf.sigmoid
        self.init_learning_rate = 1e-3
        self.domain = [0, 500]
        
        # Inputs and targets placeholders for trained data
        self.xnet = tf.placeholder(dtype=tf.float32, shape=[None, self.x.shape[1]], name='xinput')
        self.tnet = tf.placeholder(dtype=tf.float32, shape=[None, self.t.shape[1]], name='tinput')
        self.uTarget = tf.placeholder(dtype=tf.float32, shape=(None, self.output_dimension), name='target')
        
        # Define networks
        self.unet = self.uNetwork(self.xnet, self.tnet)
        self.fnet = self.fNetwork(self.xnet, self.tnet)
        
        # Define loss function and optimizer
        self.loss = tf.reduce_mean(tf.square(self.u_net - self.uTarget)) #+ tf.reduce_mean(tf.square(self.f_pred))
        self.optimizer = tf.train.AdamOptimizer()
        self.train_step = self.optimizer.minimize(self.loss)
        
        # Initialize tf variables and session
        init = tf.global_variables_initializer()
        self.sess.run(init)
   
    # General definition of network
    def neuralNetwork(self, x, t):
        """Neural network architecture"""
        # Input
        networkInput = tf.variable(tf.concat([x,t],1), dtype=tf.float32)

        # Hidden layers
        hidden = (self.num_layers-1)*[None]
        hidden[0] = networkInput
        for l in range(self.num_layers-2):
            hidden[l+1] = tf.layers.dense(hidden[l], self.neurons_per_layer, activation=self.activation)
            
        # Add prediction outermost layer
        networkPrediction = tf.layers.dense(hidden[self.num_layers-2], self.output_dimension, activation=None, name='prediction')
        return networkPrediction
    
    
    def uNetwork(self, x, t):
        u = self.neuralNetwork(x, t)
        return u
    
    def fNetwork(self, x, t):
        f = 0
        return f
    
