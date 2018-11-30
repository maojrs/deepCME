import numpy as np
import tensorflow as tf

# Neural network class specialized in solving the Chemical Master Equation (CME)
class neuralNetworkCME:

    def __init__(self, inputData, targetData, num_layers, neurons_per_layer, activation, input_dimension, output_dimension):
        """ Initializes base parameters, architecture, loss function and 
        optimizer for the neural network. It also initializes tensor flow 
        session and variables"""
        
        self.x = inputData[:,0:1]
        self.t = inputData[:,1:2]
        self.targetData = targetData
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.input_dimension = inputData.shape[1]
        self.output_dimension = np.array(targetData).shape[1]
        
        self.activation = activation 
        #self.init_learning_rate = 1e-8 # 1e-3
        self.domain = [0, 500]
        
        self.session = tf.Session()
        
        # Inputs and targets placeholders for trained data
        self.xnet = tf.placeholder(dtype=tf.float32, shape=[None, self.x.shape[1]], name='xinput')
        self.tnet = tf.placeholder(dtype=tf.float32, shape=[None, self.t.shape[1]], name='tinput')
        self.uTarget = tf.placeholder(dtype=tf.float32, shape=(None, self.output_dimension), name='target')
        
        # Define networks
        self.unet = self.uNetwork(self.xnet, self.tnet)
        self.fnet = self.fNetwork(self.xnet, self.tnet)
        
        # Define loss function, error function and optimizer
        self.loss = tf.reduce_mean(tf.square(self.unet - self.uTarget)) #+ tf.reduce_mean(tf.square(self.f_pred))
        self.error = tf.reduce_max(tf.abs(self.unet - self.uTarget))
        self.optimizer = tf.train.AdamOptimizer()
        self.train_step = self.optimizer.minimize(self.loss)
        
        # Initialize tf variables and session
        init = tf.global_variables_initializer()
        self.session.run(init)
   
   
    def neuralNetwork(self, x, t):
        """Neural network architecture"""
        # Input
        networkInput = tf.concat([x,t],1)

        # Hidden layers
        hidden = (self.num_layers-1)*[None]
        hidden[0] = networkInput
        for l in range(self.num_layers-2):
            hidden[l+1] = tf.layers.dense(hidden[l], self.neurons_per_layer, activation=self.activation)
            
        # Add prediction outermost layer
        networkPrediction = tf.layers.dense(hidden[self.num_layers-2], self.output_dimension, activation=None, name='prediction')
        return networkPrediction
    
    
    def uNetwork(self, x, t):
        ''' Network that approximates u(x,t) '''
        u = self.neuralNetwork(x, t)
        return u
    
    
    def fNetwork(self, x, t):
        ''' Network that approximates f(x,t) = u_t - Lu '''
        f = 0
        return f
    

    def train(self, num_epochs, batch_size, data_size):
        ''' Train network using stochastic gradient descent steps with Adam'''
        num_batches = int(data_size/batch_size)
        print('\nStarted training...')
        print('{:8s}\t{:8s}\t{:8s}\t{:8s}'.format('Epoch', 'Iteration', 'l2-loss', 'linf-err'))
        print('{:8s}\t{:8s}\t{:8s}\t{:8s}'.format(*(4*[8*'-'])))
        for epoch in range(num_epochs):
            # generate random batch of inputs and corresponding target values
            indicesList = np.random.choice(data_size, [num_batches, batch_size], replace = False)
            # Loop over all possible batches within the dataset
            for jbatch in range(num_batches):
                indices =  (indicesList[jbatch]).astype(int)
                inputX = np.take(self.x, indices, axis=0)
                inputT = np.take(self.t, indices, axis=0)
                targetBatch = np.take(np.array(self.targetData), indices, axis = 0)
                
                # take gradient descent step and compute loss & error
                loss_val, error_val, _ = self.session.run(
                    [self.loss, self.error, self.train_step],
                    feed_dict={self.xnet: inputX, self.tnet: inputT, self.uTarget: targetBatch}
                )
                if (epoch*num_batches + jbatch) % 500 == 0:
                    print('{:8d}\t{:8d}\t{:1.5e}\t{:1.5e}'.format(epoch, epoch*num_batches + jbatch, loss_val, error_val))
        print('...finished training.\n')
        

    def runNetwork(self, Xnew):
        ''' Run the network to obtain prediction'''
        tf_dict = {self.xnet: Xnew[:,0:1], self.tnet: Xnew[:,1:2]}
        
        unew = self.session.run(self.unet, tf_dict)
        #f_star = self.sess.run(self.fnet, tf_dict)
        
        return unew