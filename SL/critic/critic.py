import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from data import process

class Critic(Model):
    '''
    TODO:
    Critic Model
    a. Input: a state (e.g. one phase, 81 * 35 vector)
    b. Output: (7,) vector representing the value for each power.
    '''

    def __init__(self):
        '''
        Initialization for Critic Model

        Args:

        '''

        super(Critic, self).__init__()
        self.d1 = tf.keras.layers.Dense(256,activation="relu")
        self.d2 = tf.keras.layers.Dense(7)

    def call(self, states):
        '''
        Call method for critic

        Args:
        states - [batch_sz, state_size] vector where state_size = 81*35

        Returns:
        [batch_sz, 7] vector of the values for each power in each state.
        '''
        out = self.d1(states)
        out = self.d2(states)
        return out

    def loss(self, predicted_values, returns):
        """
        Args:
        predicted values - [batch_sz, 7] vector of predicted values for each power
        returns - [batch_sz, 7] vector of actual values for each power

        Returns:
        MSE(predicted, returns)
        """
        return tf.square(predicted_values - returns)
