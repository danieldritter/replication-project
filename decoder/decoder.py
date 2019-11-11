import constants
import numpy as np
import tensorflow as tf
from tf.keras import Model
from tf.keras.optimizers import Adam
from decoder.mask import masked_softmax

class Decoder(Model):
    '''
    Decoder Model
    '''

    def __init__(self):
        '''
        Initialization for Decoder Model

        Args:
        '''
        super(Model, self).__init__()

        self.h_dec_size = constants.ORDER_VOCABULARY # TODO fix
        self.optimizer = Adam(0.001)
        self.lstm = tf.keras.layers.LSTMCell(units=self.h_dec_size, activation=None) # initialize

        # define LSTM layer here

    def call(self, h_enc, mask):
        '''
        Call method for decoder

        Args:
        h_enc - the output of the encoder, an [81, H_ENC_COLS] array.
            Ordered along the 0th-dim s.t. each province is already adjacent to the next
            (so already topsorted).

        Returns:
        The concatenated encoded output of the board inputs and the previous order inputs 
        '''

        orders_list = []
        h_dec = tf.Variable(tf.random.normal([constants.NUM_PLACES, self.h_dec_size], stddev=0.1,dtype=tf.float32)) # initial decoder state - should this be a variable?
        for province in h_enc:
            # province is h^i^t_enc in paper, 
            previous_state = tf.concat(province, order)
            h_dec = self.lstm(h_dec, previous_state) 
            order = masked_softmax(h_dec, mask) # TODO: implement in mask.py
            orders_list.append(order)
            # LSTMCell stuff
                # make sure to append previous order with province (province is h^i^t_enc in paper)
            # also record orders output by each loop through (i.e. for each province)


        return orders_list

    def loss(self, probs, labels):
        return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels, probs))