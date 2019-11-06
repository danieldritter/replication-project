import constants
import numpy as np
import tensorflow as tf
from tf.keras import Model
from tf.keras.optimizers import Adam

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

        self.h_dec_size = 100 # TODO fix
        self.optimizer = Adam(0.001)
        self.lstm = tf.keras.layers.LSTMCell(units=self.h_dec_size, activation=None) # initialize

        # define LSTM layer here

    def call(self, h_enc):
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
            h_dec = self.lstm(h_dec, province) 
            order = masked_softmax(h_dec) # TODO: implement in mask.py

            # LSTMCell stuff
                # make sure to append previous order with province (province is h^i^t_enc in paper)
            # also record orders output by each loop through (i.e. for each province)


        return orders_list

    def loss():
        pass
    
def get_power_season(input):
    '''
    Function to extract the power and season of the input
    '''

    return