"""
.. module:: ann
    :synopsis: main module for artificial neural network implementation

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 9, 2017
"""
from eliana.lib.eliana_image import ElianaImage
import numpy as np
import tensorflow as tf


class ANN:

    def __init__(
            self,
            img: ElianaImage,
            epochs=2000,
            training_size=400
    ):
        self.__epochs = epochs
        self.__training_size = training_size

        self.__session = tf.InteractiveSession()
        self.__init_ann()
        self.__session.run(tf.initialize_all_variables())

    def __init_ann(self):

        self.inputs = tf.placeholder(tf.float32, shape=[None, 2])
        self.outputs = tf.placeholder(tf.float32, shape=[None, 1])

        hidden_neurons = 3

        input_weights = tf.Variable(tf.truncated_normal([2, hidden_neurons]))
        input_biases = tf.Variable(tf.zeros([hidden_neurons]))

        hidden_weights = tf.Variable(tf.truncated_normal([hidden_neurons, 3]))
        hidden_biases = tf.Variable(tf.zeros([3]))

        output_weights = tf.Variable(tf.truncated_normal([3, 1]))
        output_biases = tf.Variable(tf.zeros([1]))

        input_activation = tf.nn.sigmoid(
            tf.matmul(
                self.inputs,
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

        self.output_activation = tf.nn.sigmoid(
            tf.matmul(
                hidden_activation,
                output_weights
            ) +
            output_biases
        )

        self.error = 0.5 * tf.reduce_sum(
            tf.subtract(self.output_activation, self.outputs) *
            tf.subtract(self.output_activation, self.outputs)
        )

        learning_rate = 0.05
        self.step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.error)

        # self.__session.run(tf.initialize_all_variables())

    def train(self):

        training_size = 400
        training_inputs = [[0.0996, 0.49184], [0.2742, 0.36230]] * training_size
        training_outputs = [[0.1], [0.2]] * training_size

        epochs = 2000

        for epoch in range(epochs):
            _, error_rate = self.__session.run(
                [self.step, self.error],
                feed_dict={
                    self.inputs: np.array(training_inputs),
                    self.outputs: np.array(training_outputs)
                }

            )

            print('epoch:', str(epoch), '| error:', str((error_rate * 100)) + '%')

        print(
            self.__session.run(
                self.output_activation,
                feed_dict={
                    self.inputs: np.array([[0.0996, 0.49184]])
                }
            )
        )
        print(
            self.__session.run(
                self.output_activation,
                feed_dict={
                    self.inputs: np.array([[0.2742, 0.36230]])
                }
            )
        )


ANN(img=None).train()
