import tensorflow as tf
import numpy as np
 
# 2-D Self-Organizing Map with Gaussian Neighbourhood function and linearly decreasing learning rate
class SOM(object):
    
    #To check if the SOM has been trained
    _trained = False
 
    # Initialize
 
    # m x n are the dimensions of the SOM. 
    # dim is the dimensionality of the training inputs.
    # 'alpha' is a number denoting the initial time(iteration no)-based learning rate. Default value is 0.3
    # 'sigma' is the the initial neighbourhood value, denoting the radius of influence of the BMU while training. By default is half of max(m, n)
    def __init__(self, m, n, dim, epochs=100, alpha=None, sigma=None):
    
        # Init
        self._m = m
        self._n = n
        if alpha is None: alpha = 0.3
        else: alpha = float(alpha)
        if sigma is None: sigma = max(m, n) / 2.0
        else: sigma = float(sigma)
        self._n_iterations = abs(int(epochs))
 
        self._graph = tf.Graph()
        with self._graph.as_default():
 
            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE
 
            #Randomly initialized weightage vectors for all neurons stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal([m*n, dim]))
 
            #Matrix of size [m*n, 2] for SOM grid locations of neurons
            self._location_vects = tf.constant(np.array(list(self._neuron_locations(m, n))))
 
            ##PLACEHOLDERS FOR TRAINING INPUTS
   
            #The training vector
            self._vect_input = tf.placeholder("float", [dim])
            #Iteration number
            self._iter_input = tf.placeholder("float")
 
            ##CONSTRUCT TRAINING OP PIECE BY PIECE

            #Only the final, 'root' training op needs to be assigned as an attribute to self, since all the rest will be executed automatically during training
 
            #To compute the BMU given a vector Basically calculates the Euclidean distance between every neuron's weightage vector and the input, and returns theindex of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self._weightage_vects, tf.stack([self._vect_input for i in range(m*n)])), 2), 1)), 0)
 
            #This will extract the location of the BMU based on the BMU' index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),np.array([[0, 1]])); slice_input = tf.cast(slice_input, tf.int32)
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input, tf.constant(np.array([1, 2]))),[2])
 
            #To compute the alpha and sigma values based on iteration number
            learning_rate_op = tf.subtract(1.0, tf.div(self._iter_input, self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)
 
            #Construct the op that will generate a vector with learning rates for all neurons, based on iteration number and location wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(self._location_vects, tf.stack([bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)
 
            #Finally, the op that will use learning_rate_op to update the weightage vectors of all neurons based on a particular input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(learning_rate_op, np.array([i]), np.array([1])), [dim]) for i in range(m*n)])
            weightage_delta = tf.multiply(learning_rate_multiplier,tf.subtract(tf.stack([self._vect_input for i in range(m*n)]), self._weightage_vects))                                         
            new_weightages_op = tf.add(self._weightage_vects, weightage_delta)
            self._training_op = tf.assign(self._weightage_vects, new_weightages_op)                                       
 
            ##INITIALIZE SESSION
            self._sess = tf.Session()
 
            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)
 
    # Yields the 2-D locations of the individual neurons in the SOM. Nested iterations over both dimensions to generate all 2-D locations in the map
    def _neuron_locations(self, m, n):
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
 
    # Training: input_vects should be an iterable of 1-D array with dim as provided during init of the SOM. Current weight vectors for all neurons(initially random) are taken as sta
    def train(self, input_vects):
        for iter_no in range(self._n_iterations):
            #Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})
 
        #Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid
 
        self._trained = True
 
    # Returns a list of 'm' lists, with each inner list containing the 'n' corresponding centroid locations as 1-D NumPy arrays.
    def get_centroids(self):
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid
 
    # Maps each input vector to the relevant neuron in the SOM grid. 'input_vects' should be an iterable of 1-D NumPy arrays with dimensionality as provided during initialization of this SOM.
    # Returns a list of 1-D NumPy arrays containing (row, column) info for each input vector(in the same order), corresponding to mapped neuron.
    def map_vects(self, input_vects):
        if not self._trained:
            raise ValueError("SOM not trained yet")
 
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))], key=lambda x: np.linalg.norm(vect - self._weightages[x]))
            to_return.append(self._locations[min_index])
 
        return to_return

