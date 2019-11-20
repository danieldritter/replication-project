from constants.constants import ORDER_VOCABULARY_SIZE, NUM_PLACES
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from SL.decoder.mask import masked_softmax

class Decoder(Model):
    '''
    Decoder Model
    '''

    def __init__(self):
        '''
        Initialization for Decoder Model

        Args:
        '''
        super(Decoder, self).__init__()

        self.h_dec_size = ORDER_VOCABULARY_SIZE

        # LSTM layer here
        self.lstm = tf.keras.layers.LSTMCell(units=self.h_dec_size, activation=None) # initialize


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

        # accumulating taken orders
        orders_list = []

        # initializing initial state passed to decoder, which is the size of the output of the encoder (num_phases x ORDER_VOCABULARY_SIZE)
        h_dec = tf.random.normal([int(h_enc.shape[0]), self.h_dec_size], stddev=0.1, dtype=tf.float32)

        for province in tf.unstack(h_enc, axis=1):
            # province is h^i^t_enc in paper,
            # valid_orders = masked_softmax(h_dec, mask) # TODO: implement in mask.py

            valid_orders = h_dec
            # make sure to append previous order with province (province is h^i^t_enc in paper)
            # also record orders output by each loop through (i.e. for each province)
            previous_state = tf.concat((province, valid_orders), axis=1)
            print(previous_state.shape)
            print(valid_orders.shape)
            h_dec, out = self.lstm(h_dec, [province, valid_orders])
            print("TEST")
            orders_list.append(valid_orders)

        return orders_list
