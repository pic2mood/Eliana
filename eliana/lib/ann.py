"""
.. module:: ann
    :synopsis: main module for artificial neural network implementation

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 9, 2017
"""
from eliana.lib.eliana_image import ElianaImage
import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod


class ANNComponent(ABC):
    @abstractmethod
    def init(self):
        pass


class Neuron(ANNComponent):

    def __init__(self, shape=[]):

        input_count, output_count = shape

        self.input = self.init(input_count)
        self.output = self.init(output_count)

    def init(self, count):
        return tf.placeholder(
            tf.float32, shape=[None, count]
        )


class Weight(ANNComponent):

    def __init__(self, shape=[]):
        i, h, o = shape
        self.input = self.init([i, h])
        self.hidden = self.init([h, h])
        self.output = self.init([h, o])

    def init(self, weight):
        return tf.Variable(tf.truncated_normal(weight))


class Bias(ANNComponent):

    def __init__(self, shape=[]):
        h, o = shape
        self.input = self.init(h)
        self.hidden = self.init(h)
        self.output = self.init(o)

    def init(self, bias):
        return tf.Variable(tf.zeros(bias))


class Activation(ANNComponent):

    def __init__(self, inputs, weights, bias):

        self.input = self.init(inputs, weights.input, bias.input)
        self.hidden = self.init(self.input, weights.hidden, bias.hidden)
        self.output = self.init(self.hidden, weights.output, bias.output)

    def init(self, source, weight, bias):
        return tf.nn.sigmoid(
            tf.matmul(
                source,
                weight
            ) +
            bias
        )


class Optimizer(ANNComponent):

    def __init__(self, activation, neurons, error, learning_rate):
        self.optimizer = self.init(activation, neurons, error, learning_rate)

    def init(self, activation, neurons, error, learning_rate):
        error = error * tf.reduce_sum(
            tf.subtract(activation.output, neurons.output) *
            tf.subtract(activation.output, neurons.output)
        )

        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate
        ).minimize(
            error
        )

        return error, optimizer


class ANN:

    def __init__(
        self,
        shape=[2, 3, 1],
        model=''
    ):

        self.__model = model
        self.__session = tf.InteractiveSession()

        self.__init_ann(shape)
        self.__session.run(tf.initialize_all_variables())
        self.__saver = tf.train.Saver()

    def __init_ann(self, shape=[]):

        input_count, hidden_count, output_count = shape

        self.__neurons = Neuron(
            shape=[
                input_count, output_count
            ]
        )
        weights = Weight(
            shape=[
                input_count,
                hidden_count,
                output_count
            ]
        )
        bias = Bias(
            shape=[
                hidden_count,
                output_count
            ]
        )
        self.__activation = Activation(
            self.__neurons.input,
            weights,
            bias
        )
        self.__error, self.__step = Optimizer(
            self.__activation,
            self.__neurons,
            error=0.5,
            learning_rate=0.05
        ).optimizer

    def train(self, epochs=2000, training_size=400):

        training_inputs = [[0.0996, 0.49184], [0.2742, 0.36230]] * training_size
        training_outputs = [[0.1], [0.2]] * training_size

        for epoch in range(epochs):
            _, error_rate = self.__session.run(
                [self.__step, self.__error],
                feed_dict={
                    self.__neurons.input: np.array(training_inputs),
                    self.__neurons.output: np.array(training_outputs)
                }

            )

            print('epoch:', str(epoch), '| error:', str((error_rate * 100)) + '%')

        # print(
        #     self.__session.run(
        #         self.__output_activation,
        #         feed_dict={
        #             self.__inputs: np.array([[0.0996, 0.49184]])
        #         }
        #     )
        # )
        # print(
        #     self.__session.run(
        #         self.__output_activation,
        #         feed_dict={
        #             self.__inputs: np.array([[0.2742, 0.36230]])
        #         }
        #     )
        # )

    def save(self):
        self.__saver.save(self.__session, self.__model)

    def run(self, img: ElianaImage):

        self.__saver.restore(self.__session, self.__model)

        result = self.__session.run(
            self.__activation.output,
            feed_dict={
                # self.__neurons.input: np.array([[img.texture, img.colorfulness]])
                self.__neurons.input: np.array([[0.2742, 0.36230]])
            }
        )

        print('ANN result:', result)


import os

model_path = os.path.join(
    os.getcwd(),
    'training',
    'models',
    'eliana_ann_overall',
    'eliana_ann_overall'
)

a = ANN(model=model_path)
a.train()
a.save()
a.run(None)
