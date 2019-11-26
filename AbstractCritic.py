import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from constants.constants import UNIT_POWER, STATE_SIZE, NUM_POWERS

class AbstractCritic(Model):
    '''
    Critic Model
    a. Input: a state (e.g. one phase, 81 * 35 vector)
    b. Output: (7,) vector representing the value for each power.
    '''

    def __init__(self):
        '''
        Initialization for Critic Model

        Args:
        '''
        super(AbstractCritic, self).__init__()
        self.crit1 = tf.keras.layers.Dense(256, input_shape=(-1, STATE_SIZE), activation="relu", dtype=tf.float32)
        self.crit2 = tf.keras.layers.Dense(NUM_POWERS, dtype=tf.float32)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def call(self, states):
        '''
        Call method for critic

        Args:
        states - [batch_sz, state_size] vector where state_size = 81*35
        
        Returns:
        [batch_sz, 7] vector of the values for each power in each state.
        '''
        out1 = self.crit1(states)
        values = self.crit2(out1)
        return values


    def loss(self, predicted_values, returns):
        """
        Args:
        predicted values - [batch_sz, 7] vector of predicted values for each power
        returns - [batch_sz, 7] vector of actual values for each power
        
        Returns:
        Loss
        """
        raise NotImplementedError("Not implemented in abstract class.")


