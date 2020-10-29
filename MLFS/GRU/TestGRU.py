import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from GRU import *
from GRUUtils import *




# 1. Dataset creation (100 examples of numbers represented in 5 bits)
batchSize = 100
timeSize = 5
Xtrain, Ytrain = GenBatch(timeSize, batchSize)
Xtest, Ytest = GenBatch(timeSize, batchSize)

# 2. Create Model
inpDim = 2 # for each term 
hidDim = 16


gru = GRU(inpDim, hidDim)
Wout = tf.Variable(tf.truncated_normal(dtype=tf.float64, shape=(hidDim, 1), mean=0, stddev=0.01))
bout = tf.Variable(tf.truncated_normal(dtype=tf.float64, shape=(1,), mean=0, stddev=0.01))
output = tf.map_fn(lambda h_t: tf.matmul(h_t, Wout) + bout, gru.h_t)

# Expected output & loss
expOutput = tf.placeholder(dtype=tf.float64, shape=(batchSize, timeSize, 1), name='expected_output')
loss = tf.reduce_sum(0.5 * tf.pow(output - expOutput, 2)) / float(batchSize)

trainStep = tf.train.AdamOptimizer().minimize(loss)

# Initialize
session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)
trainLoss = []
validLoss = []

# 2. Train
for epoch in range(5000):
    # Compute the losses
    _, train_loss = session.run([trainStep, loss], feed_dict={gru.inpLay: Xtrain, expOutput: Ytrain})
    validation_loss = session.run(loss, feed_dict={gru.inpLay: Xtest, expOutput: Ytest})
    
    # Log the losses
    trainLoss += [train_loss]
    validLoss += [validation_loss]
    
    # Display an update every 50 iterations
    if epoch % 50 == 0:
        plt.plot(trainLoss, '-b', label='Train loss')
        plt.plot(validLoss, '-r', label='Validation loss')
        plt.legend(loc=0)
        plt.title('Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.show()
        print('Iteration: %d, train loss: %.4f, test loss: %.4f' % (epoch, train_loss, validation_loss))
    

    
# Define two numbers a and b and let the model compute a + b
a = 1024
b = 16

# The model is independent of the sequence length! Now we can test the model on even longer bitstrings
bitstring_length = 20

# Create the feature vectors    
X_custom_sample = np.vstack([AsBytes(a, bitstring_length), AsBytes(b, bitstring_length)]).T
X_custom = np.zeros((1,) + X_custom_sample.shape)
X_custom[0, :, :] = X_custom_sample

# Make a prediction by using the model
y_predicted = session.run(output, feed_dict={gru.inpLay: X_custom})
# Just use a linear class separator at 0.5
y_bits = 1 * (y_predicted > 0.5)[0, :, 0]
# Join and reverse the bitstring
y_bitstr = ''.join([str(int(bit)) for bit in y_bits.tolist()])[::-1]
# Convert the found bitstring to a number
y = int(y_bitstr, 2)

# Print out the prediction
print(y) # Yay! This should equal 1024 + 16 = 1040

