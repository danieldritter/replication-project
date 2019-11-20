import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam

import process
from SL.encoder.encoder import Encoder
from SL.decoder.decoder import Decoder
from constants.constants import SEASON, UNIT_POWER

class SL_model(Model):
    '''
    The supervised learning model for the Diplomacy game
    '''

    def __init__(self, num_board_blocks, num_order_blocks):
        '''
        Initialization for Encoder Model

        Args:
        num_board_blocks - number of blocks for encoding the board state
        num_order_blocks - number of blocks for encoding previous orders
        '''

        super(Model, self).__init__()

        # creating encoder and decoder networks
        self.encoder = Encoder(num_board_blocks, num_order_blocks)
        self.decoder = Decoder()

    def call(self, state_inputs, order_inputs, power_season):
        '''
        Function to run the SL model

        Keyword Args:
        state_inputs - the board state inputs
        order_inputs - the previous order inputs
        power_season - the power and season to be used in film

        Returns:
        a probability distribution over valid orders
        '''

        # casting inputs to float32
        state_inputs = tf.cast(state_inputs, tf.float32)
        order_inputs = tf.cast(order_inputs, tf.float32)
        
        enc_out = self.encoder.call(state_inputs, order_inputs, power_season)
        dec_out = self.decoder.call(enc_out, None)
        return dec_out

    def loss(self, probs, labels):  
        '''
        Function to compute the loss of the SL Model

        Keyword Args:
        probs - the probability distribution output over the orders
        labels - the labels representing the actions taken

        Return:
        the crossentropy loss for the actions taken
        '''
        return tf.reduce_sum(sparse_categorical_crossentropy(labels, probs))