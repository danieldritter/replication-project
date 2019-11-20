from constants.constants import H_ENC_COLS, NUM_PLACES  
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

        self.h_dec_size = [NUM_PLACES, H_ENC_COLS]

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
        
        # initializing initial state passed to decoder
        h_dec = tf.Variable(tf.random.normal(self.h_dec_size, stddev=0.1,dtype=tf.float32)) # initial decoder state - should this be a variable?
        
        for province in tf.unstack(h_enc, axis=0):
            # province is h^i^t_enc in paper, 
            # valid_orders = masked_softmax(h_dec, mask) # TODO: implement in mask.py 
            valid_orders = h_dec

            # make sure to append previous order with province (province is h^i^t_enc in paper)
            # also record orders output by each loop through (i.e. for each province)
            # print(tf.concat((province, valid_orders), axis=1).shape)
            # print(h_dec.shape)
            previous_state = (province, valid_orders)
            h_dec = self.lstm(h_dec, previous_state) 
            orders_list.append(valid_orders)

        return orders_list