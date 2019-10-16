import tensorflow as tf 
from tf.keras.layers import ReLU
from tf.keras.layers import BatchNormalization
from tf.keras.layers import Layer

from film import FiLM

class Block(Layer):
    '''
    A block layer consisting of the GCN, batch normalization, FiLM, 
    and ReLU
    '''

    def __init__(self):
        '''
        Initializer for a Block object
        '''
        
        self.gcn = GCN()
        self.bn = BatchNormalization(axis=1, epsilon=0.0001)
        self.film = FiLM()
        self.relu = ReLU()

    def call(self, block_input, is_training=True):
        '''
        Method to pass inputs through a block

        Args:
        inputs - input data
        '''
        
        gcn_out = self.gcn(block_input)
        yblo = self.bn(gcn_out, training=is_training)
        
        # something about power and season?
        gamma, beta = film(normalized)
        zlbo = gamma * yblo + beta

        out = self.relu(zlbo) + block_input
        return out




