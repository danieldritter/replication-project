import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, ReLU, BatchNormalization, Dense
from SL.encoder import film
from constants import constants

class Block(Layer):
    '''
    A block layer consisting of the GCN, batch normalization, FiLM,
    and ReLU
    '''

    def __init__(self, input_size):
        '''
        Initializer for a Block object
        
        Keyword Args:
        input_size - size of inputs (81 * 35) for board and (81 * 40) for orders
        '''

        super(Block,self).__init__()

        self.gcn = GCN(input_size)
        self.bn = BatchNormalization(axis=1, epsilon=0.0001)
        self.film = film.FiLM()
        self.relu = ReLU()

    def call(self, block_input, power_season, is_training=True):
        '''
        Method to pass inputs through a block layer (graph convolutional layer, 
        batch norm, film, ReLU)

        Keyword Args:
        block_input - the inputs to the block (either board state or prev orders)
        power_season - the current power and season of the game (used in FiLM)

        Returns:
        the output of the embedding
        '''

        gcn_out = self.gcn(block_input)
        ylbo = self.bn(gcn_out, training=is_training)

        # something about power and season?
        gamma, beta = self.film(power_season)
        # print(ylbo.shape)
        # print(gamma.shape)
        # print(beta.shape)
        zlbo = gamma[:, :, None] *  ylbo + beta[:, :, None] # increasing dimension for elemntwise mult and add
        out = self.relu(zlbo) + block_input # adding residual connection
        return out

class GCN(Layer):
    '''
    Class defining a layer of the graph convolutional network
    '''

    def __init__(self, input_size):
        '''
        Initializer for a Block object

        Keyword Args:
        input_size - the size of the inputs (81 * 35) for board state or (81 * 40) for prev order
        '''

        super(GCN,self).__init__()

        self.input_size = input_size
        self.d1 = Dense(input_size)

    def call(self, inputs): 
        '''
        Function call for GCN, applying the adjacency matrix and a dense layer
        '''

        # applying adjacency matrix as floats
        # print(inputs.shape)
        inputs = tf.tensordot(inputs, constants.A.astype(np.float32), axes=[[1], [0]]) # cant figure out shape issues using reshape to fix
        inputs = tf.reshape(inputs, (inputs.shape[0], 81, self.input_size))
        gcn_out = self.d1(inputs)
        return gcn_out
