import tensorflow as tf
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Layer

from . import film

class Block(Layer):
    '''
    A block layer consisting of the GCN, batch normalization, FiLM,
    and ReLU
    '''

    def __init__(self):
        '''
        Initializer for a Block object
        '''

        self.gcn = lambda x: x
        self.bn = BatchNormalization(axis=1, epsilon=0.0001)
        self.film = film.FiLM()
        self.relu = ReLU()

    def call(self, block_input, power_season, is_training=True):
        '''
        Method to pass inputs through a block

        Args:
        inputs - input data
        power_season - the current power and season of the game (used in FiLM)
        '''

        gcn_out = self.gcn(block_input)
        yblo = self.bn(gcn_out, training=is_training)

        # something about power and season?
        gamma, beta = self.film(power_season)
        zlbo = gamma * yblo + beta

        out = self.relu(zlbo) + block_input
        return out
