
# neural network implementation in script mode

import numpy as np
import tensorflow as tf

session = tf.InteractiveSession()


# 1. setup neural network component sizes
#   2 input neurons
#       1 for Color,
#       1 for Texture

inputs = tf.placeholder(tf.float32, shape=[None, 2])

# 1 output neurons

outputs = tf.placeholder(tf.float32, shape=[None, 1])

# 2 hidden neurons

hidden_neurons = 2

# 2. initialize neural network component values
# input layer

input_weights = tf.Variable(tf.truncated_normal([2, hidden_neurons]))
input_biases = tf.Variable(tf.zeros([hidden_neurons]))

# hidden layer

hidden_weights = tf.Variable(tf.truncated_normal([hidden_neurons, 2]))
hidden_biases = tf.Variable(tf.zeros([2]))

# output layer

output_weights = tf.Variable(tf.truncated_normal([2, 1]))
output_biases = tf.Variable(tf.zeros([1]))

# 3. setup neural network functions
# activation function

input_activation = tf.nn.sigmoid(
    tf.matmul(
        inputs,
        input_weights
    ) +
    input_biases
)

hidden_activation = tf.nn.sigmoid(
    tf.matmul(
        input_activation,
        hidden_weights
    ) +
    hidden_biases
)

output_activation = tf.nn.sigmoid(
    tf.matmul(
        hidden_activation,
        output_weights
    ) +
    output_biases
)

# optimizer

error = 0.5 * tf.reduce_sum(
    tf.subtract(output_activation, outputs) *
    tf.subtract(output_activation, outputs)
)

learning_rate = 0.005
step = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

# 4. build the network
session.run(tf.initialize_all_variables())

# 5. train
training_size = 300
training_inputs = 
training_output = 


