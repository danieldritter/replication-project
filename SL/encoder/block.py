import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, ReLU, BatchNormalization, Dense
from SL.encoder import film
from constants import constants

class FirstBlock(Layer):
    '''
    A block layer consisting of the GCN, batch normalization, FiLM,
    and ReLU
    '''

    def __init__(self, input_size):
        '''
        Initializer for a FirstBlock object

        Keyword Args:
        input_size - size of inputs (81 * 35) for board and (81 * 40) for orders
        '''

        super(FirstBlock,self).__init__()

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
        # reshaping power and season then inputting
        power_season_input = tf.reshape(power_season,(power_season.shape[0],power_season.shape[1],1))
        gamma, beta = self.film(power_season_input, power_season)
        zlbo = tf.expand_dims(gamma,axis=2)*ylbo + tf.expand_dims(beta,axis=2) # increasing dimension for elemntwise mult and add
        out = self.relu(zlbo) # no residual connection!!!
        return out

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

        # reshaping power and season then inputting
        power_season_input = tf.reshape(power_season,(power_season.shape[0],power_season.shape[1],1))
        gamma, beta = self.film(power_season_input, power_season)
        zlbo = tf.expand_dims(gamma,axis=2)*ylbo + tf.expand_dims(beta,axis=2) # increasing dimension for elemntwise mult and add
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
        self.A_matrix = tf.convert_to_tensor(constants.A.astype(np.float32))

    def call(self, inputs):
        '''
        Function call for GCN, applying the adjacency matrix and a dense layer
        '''

        # applying adjacency matrix as floats
        # inputs = tf.tensordot(inputs, self.A_matrix, axes=[[1], [0]])
        inputs = tf.einsum('ijk,j...->ijk',inputs,self.A_matrix)
        # inputs = tf.transpose(inputs, (0,2,1))
        gcn_out = self.d1(inputs)
        return gcn_out
